import argparse
import argbind
from functools import partial
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

torch.set_float32_matmul_precision("high")

from realchords.utils.lr_scheduler import LinearWarmupCosineDecay
from realchords.rl.deepspeed import get_strategy
from realchords.rl.openrlhf_local import AdaptiveKLController, FixedKLController

from realchords.lit_module.decoder_only import LitDecoder
from realchords.lit_module.enc_dec import LitEncoderDecoder
from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.lit_module.discriminative_reward import LitDiscriminativeReward
from realchords.lit_module.contrastive_reward_rhythm import (
    LitContrastiveRewardRhythm,
)
from realchords.lit_module.discriminative_reward_rhythm import (
    LitDiscriminativeRewardRhythm,
)
from realchords.utils.inference_utils import load_lit_model
from realchords.rl.utils import ModelPreparer
from realchords.rl.actor import (
    DecoderSingleAgentActor,
    EncoderDecoderOfflineAnchor,
)
from realchords.rl.critic import Critic
from realchords.rl.rl_trainer import ReaLchordsPPOTrainer
from realchords.rl.reward.model_based_rewards import (
    ContrastiveRewardFn,
    DiscriminativeRewardFn,
    ContrastiveRewardRhythmFn,
    DiscriminativeRewardRhythmFn,
)
from realchords.rl.reward.rule_based_rewards import (
    EarlyStopPenalty,
    SilencePenalty,
    LongNotePenalty,
    InvalidOutputPenalty,
    RepetitionPenalty,
)
from realchords.rl.experience_maker import ExperienceMaker
from realchords.utils.train_utils import AttrDict


@argbind.bind(without_prefix=True)
def main(args, save_dir: str = ""):
    if not save_dir:
        raise ValueError("save_dir must be provided.")
    args.save_dir = save_dir
    args.wandb_run_name = Path(save_dir).name
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model, tokenizer, dataloaders = load_lit_model(
        model_path=args.pretrain_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.rollout_batch_size,
        # Disable dropout in RL fine-tuning
        override_args={
            "DecoderTransformer.dropout": 0.0,
            **args.lit_module_override_args,
        },
    )
    train_dataloader, val_dataloader = dataloaders

    actor = DecoderSingleAgentActor(
        model,
        tokenizer=tokenizer,
        model_part=args.model_part,
        max_seq_len=model.max_seq_len - 2,  # -2 for bos and eos
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    # Remember to deepcopy the model for initialize critic
    critic = Critic(deepcopy(model))

    strategy.print(actor)
    strategy.print(critic)

    # load weights for reference actor
    anchor_model, _, _ = load_lit_model(
        model_path=args.anchor_model_path,
        lit_module_cls=LitEncoderDecoder,
        batch_size=args.rollout_batch_size,
        # Disable dropout in RL fine-tuning
        override_args={
            "EncoderDecoderTransformer.dropout": 0.0,
            **args.lit_module_override_args,
        },
    )
    initial_model = EncoderDecoderOfflineAnchor(
        anchor_model,
        bos_token_id=tokenizer.bos_token,
        eos_token_id=tokenizer.eos_token,
        pad_token_id=tokenizer.pad_token,
        max_seq_len=model.max_seq_len // 2 + 1,  # doesn't matter
    )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor,
        lr=args.actor_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2,
    )
    critic_optim = strategy.create_optimizer(
        critic,
        lr=args.critic_learning_rate,
        betas=args.adam_betas,
        weight_decay=args.l2,
    )

    # configure scheduler
    num_backward_per_episode = args.rollout_batch_size // args.train_batch_size
    max_steps = args.num_steps * num_backward_per_episode
    if isinstance(args.warmup_steps, float):
        num_warmup_steps = math.ceil(max_steps * args.warmup_steps)
    elif isinstance(args.warmup_steps, int):
        num_warmup_steps = args.warmup_steps * num_backward_per_episode
    else:
        raise ValueError("warmup_steps must be either float or int.")

    actor_scheduler = LinearWarmupCosineDecay(
        optimizer=actor_optim,
        warmup_iters=num_warmup_steps,
        total_iters=max_steps,
        eta_min=args.actor_learning_rate * 0.1,
    )

    critic_scheduler = LinearWarmupCosineDecay(
        optimizer=critic_optim,
        warmup_iters=num_warmup_steps,
        total_iters=max_steps,
        eta_min=args.critic_learning_rate * 0.1,
    )

    contrastive_reward_models = [
        load_lit_model(
            model_path,
            lit_module_cls=LitContrastiveReward,
            return_only_model=True,
        )
        for model_path in args.contrastive_reward_model_path
    ]
    discriminative_reward_models = [
        load_lit_model(
            model_path,
            lit_module_cls=LitDiscriminativeReward,
            return_only_model=True,
        )
        for model_path in args.discriminative_reward_model_path
    ]

    contrastive_reward_rhythm_models = [
        load_lit_model(
            model_path,
            lit_module_cls=LitContrastiveRewardRhythm,
            return_only_model=True,
        )
        for model_path in args.contrastive_reward_rhythm_model_path
    ]
    discriminative_reward_rhythm_models = [
        load_lit_model(
            model_path,
            lit_module_cls=LitDiscriminativeRewardRhythm,
            return_only_model=True,
        )
        for model_path in args.discriminative_reward_rhythm_model_path
    ]

    # Prepare models/optimizers using ModelPreparer
    preparer = ModelPreparer(strategy)
    preparer.add_trainable("actor", actor, actor_optim, actor_scheduler)
    preparer.add_trainable("critic", critic, critic_optim, critic_scheduler)
    preparer.add_model("initial_model", initial_model)
    preparer.add_model_list("contrastive_rewards", contrastive_reward_models)
    preparer.add_model_list("discriminative_rewards", discriminative_reward_models)
    preparer.add_model_list("contrastive_rhythm_rewards", contrastive_reward_rhythm_models)
    preparer.add_model_list("discriminative_rhythm_rewards", discriminative_reward_rhythm_models)

    models = preparer.prepare(is_rlhf=True)

    # Extract prepared models
    actor, actor_optim, actor_scheduler = models.get_trainable("actor")
    critic, critic_optim, critic_scheduler = models.get_trainable("critic")
    initial_model = models.get_model("initial_model")
    contrastive_reward_models = models.get_model_list("contrastive_rewards")
    discriminative_reward_models = models.get_model_list("discriminative_rewards")
    contrastive_reward_rhythm_models = models.get_model_list("contrastive_rhythm_rewards")
    discriminative_reward_rhythm_models = models.get_model_list("discriminative_rhythm_rewards")

    reward_configs = []
    for i, contrastive_reward_model in enumerate(contrastive_reward_models):
        reward_configs.append(
            {
                "reward_fn": ContrastiveRewardFn(
                    model=contrastive_reward_model,
                    pad_token_id=tokenizer.pad_token,
                    bos_token_id=tokenizer.bos_token,
                    eos_token_id=tokenizer.eos_token,
                    model_part=args.model_part,
                ),
                "weight": getattr(args, "contrastive_reward_weight", 1.0),
                "name": f"contrastive_reward_{i}",
            }
        )
    for i, discriminative_reward_model in enumerate(
        discriminative_reward_models
    ):
        reward_configs.append(
            {
                "reward_fn": DiscriminativeRewardFn(
                    model=discriminative_reward_model,
                    pad_token_id=tokenizer.pad_token,
                    bos_token_id=tokenizer.bos_token,
                    eos_token_id=tokenizer.eos_token,
                    model_part=args.model_part,
                ),
                "weight": getattr(args, "discriminative_reward_weight", 1.0),
                "name": f"discriminative_reward_{i}",
            }
        )

    for i, contrastive_reward_rhythm_model in enumerate(
        contrastive_reward_rhythm_models
    ):
        reward_configs.append(
            {
                "reward_fn": ContrastiveRewardRhythmFn(
                    model=contrastive_reward_rhythm_model,
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                ),
                "weight": args.contrastive_reward_rhythm_weight,
                "name": f"contrastive_reward_rhythm_{i}",
            }
        )

    for i, discriminative_reward_rhythm_model in enumerate(
        discriminative_reward_rhythm_models
    ):
        reward_configs.append(
            {
                "reward_fn": DiscriminativeRewardRhythmFn(
                    model=discriminative_reward_rhythm_model,
                    tokenizer=tokenizer,
                    model_part=args.model_part,
                ),
                "weight": args.discriminative_reward_rhythm_weight,
                "name": f"discriminative_reward_rhythm_{i}",
            }
        )

    reward_configs.append(
        {
            "reward_fn": EarlyStopPenalty(
                pad_token_id=tokenizer.pad_token,
                bos_token_id=tokenizer.bos_token,
                eos_token_id=tokenizer.eos_token,
            ),
            "weight": getattr(args, "early_stop_penalty_weight", 1.0),
            "name": "early_stop_penalty",
        },
    )
    reward_configs.append(
        {
            "reward_fn": InvalidOutputPenalty(
                tokenizer=tokenizer,
                model_part=args.model_part,
            ),
            "weight": args.invalid_output_penalty_weight,
            "name": "invalid_output_penalty",
        },
    )
    reward_configs.append(
        {
            "reward_fn": SilencePenalty(
                tokenizer=tokenizer,
            ),
            "weight": getattr(args, "silence_penalty_weight", 1.0),
            "name": "silence_penalty",
        }
    )
    reward_configs.append(
        {
            "reward_fn": LongNotePenalty(
                tokenizer=tokenizer,
            ),
            "weight": getattr(args, "long_note_penalty_weight", 0.0),
            "name": "long_note_penalty",
        }
    )
    reward_configs.append(
        {
            "reward_fn": RepetitionPenalty(
                tokenizer=tokenizer,
                model_part=args.model_part,
                threshold=getattr(args, "repetition_penalty_threshold", 4),
            ),
            "weight": getattr(args, "repetition_penalty_weight", 0.0),
            "name": "repetition_penalty",
        }
    )

    init_kl_coef = args.init_kl_coef
    kl_target = args.kl_target
    kl_horizon = args.kl_horizon
    if args.kl_target:
        kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
    else:
        kl_ctl = FixedKLController(init_kl_coef)
    experience_maker = ExperienceMaker(
        actor=actor,
        critic=critic,
        reward_configs=reward_configs,
        initial_model=initial_model,
        tokenizer=tokenizer,
        kl_controller=kl_ctl,
        strategy=strategy,
        reward_vram_swap=args.reward_vram_swap,
        logits_vram_swap=args.logits_vram_swap,
    )

    ema_model = None

    os.makedirs(save_dir, exist_ok=True)

    # configure Trainer
    trainer = ReaLchordsPPOTrainer(
        strategy,
        experience_maker=experience_maker,
        kl_ctl=kl_ctl,
        actor=actor,
        critic=critic,
        initial_model=initial_model,
        ema_model=ema_model,
        actor_optim=actor_optim,
        critic_optim=critic_optim,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        max_norm=args.max_norm,
        limit_eval_batches=args.limit_eval_batches,
        max_log_examples=args.max_log_examples,
        counterpart_prediction_loss_coef=getattr(
            args, "counterpart_prediction_loss_coef", 0.0
        ),
        # for GPT generation
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token,
        eos_token_id=tokenizer.eos_token,
        buffer_cpu_offload=args.buffer_cpu_offload,
        dataloader_pin_memory=args.dataloader_pin_memory,
        trainer_empty_cache=args.trainer_empty_cache,
    )

    trainer.fit(
        args,
        train_dataloader,
        val_dataloader,
    )

    # save model checkpoint after fitting on only rank0
    if strategy.is_rank_0():
        torch.save(actor.state_dict(), Path(save_dir) / "actor.pth")
        if args.save_value_network:
            torch.save(critic.state_dict(), Path(save_dir) / "critic.pth")


if __name__ == "__main__":
    args = argbind.parse_args()
    argbind.dump_args(args, Path(args["save_dir"]) / "args.yml")
    with argbind.scope(args):
        args = AttrDict(args)
        main(args)
