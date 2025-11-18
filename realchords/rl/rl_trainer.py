"""RL trainer class modified from the original OpenRLHF trainer."""

import os
import copy
from typing import Any, Callable, List, Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import numpy as np

from realchords.rl.openrlhf_local import (
    PPOTrainer,
    Actor,
    GPTLMLoss,
    PolicyLoss,
    ValueLoss,
    masked_mean,
    unpacking_samples,
    compute_approx_kl,
    DistributedSampler,
    pad_sequences,
    unpad_sequences,
    AdaptiveKLController,
    FixedKLController,
)

from lightning.pytorch.utilities import grad_norm
from realchords.rl.experience_maker import ExperienceMaker, Experience
from realchords.rl.replay_buffer import RealchordsReplayBuffer
from realchords.rl.utils import compute_full_kl, compute_entropy
from realchords.utils.log_utils import midi_to_audio_image
from realchords.utils.eval_utils import (
    evaluate_note_in_chord_ratio,
    evaluate_initial_silence,
    evaluate_average_duration,
)
from realchords.constants import MIDI_SYNTH_SR
from realchords.utils.logging_utils import logger

compute_entropy = torch.compile(compute_entropy)


class ReaLchordsBasePPOTrainer(PPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) algorithm.

    This class is a modified version of the OpenRLHF PPOTrainer class.
    It removed the following features:
    1. EMA
    2. Remote reward model
    3. ptx loss
    4. pretrain dataloader
    5. Aux loss (for MoE)

    It added the following features:
    1. Allow external configuration of experience maker and input as a argument.
    2. Allow a generic train_dataloader rather than prompt_dataloader.
        train_dataloader should have batch size of micro_rollout_batch_size.
    3. Allow evaluate phase and logging of evaluation results.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        experience_maker: ExperienceMaker,
        kl_ctl: AdaptiveKLController | FixedKLController,
        actor: Actor,
        critic: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        limit_eval_batches: Optional[int] = None,
        max_log_examples: int = 8,
        log_samples: bool = True,
        trainer_empty_cache: bool = True,
        counterpart_prediction_loss_coef: float = 0.0,
        **generate_kwargs,
    ) -> None:
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.micro_train_batch_size = micro_train_batch_size
        self.prompt_max_len = prompt_max_len
        self.gradient_checkpointing = gradient_checkpointing
        self.limit_eval_batches = limit_eval_batches
        self.max_log_examples = max_log_examples
        self.log_samples = log_samples
        self.trainer_empty_cache = trainer_empty_cache
        self.counterpart_prediction_loss_coef = counterpart_prediction_loss_coef

        self.actor = actor
        self.critic = critic
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()  # Disabled pretrain loss
        self.aux_loss = False  # Disabled auxiliary loss
        self.pretrain_dataloader = None  # Disabled pretrain dataloader

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        self.experience_maker = experience_maker
        self.kl_ctl = kl_ctl
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = RealchordsReplayBuffer(
            micro_train_batch_size,
            buffer_limit,
            buffer_cpu_offload,
            packing_samples,
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                dir=strategy.args.save_dir,
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True
            )
            # Change eval x-axis to global_step instead of epoch
            wandb.define_metric("eval/global_step")
            wandb.define_metric(
                "eval/*", step_metric="eval/global_step", step_sync=True
            )

        # Initialize TensorBoard writer if wandb is not available
        if (
            self.strategy.args.use_tensorboard
            and self._wandb is None
            and self.strategy.is_rank_0()
        ):
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.strategy.args.use_tensorboard, strategy.args.wandb_run_name
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def evaluate(self, dataloader, global_step):
        if self.limit_eval_batches is not None:
            limit_eval_batches = self.limit_eval_batches
        else:
            limit_eval_batches = len(dataloader)
        pbar = tqdm(
            range(limit_eval_batches),
            desc=f"Eval at step {global_step}",
            disable=not self.strategy.is_rank_0(),
        )

        num_batches = 0
        experience_to_log = None
        batch_to_log = None
        info_to_log = []
        for batch in dataloader:
            for i, experience in enumerate(
                self.experience_maker.make_experience_list(
                    batch, is_eval=True, **self.generate_kwargs
                )
            ):
                if i == 0:
                    experience_to_log = experience
                    batch_to_log = batch
                info_to_log.append(experience.info)
            pbar.update()
            num_batches += 1
            if num_batches >= limit_eval_batches:
                break

        if self.log_samples:
            self.log_eval(experience_to_log, batch_to_log, info_to_log, global_step)

    def log_eval(self, experience_to_log, batch_to_log, info_list, global_step):
        raise NotImplementedError("log_eval method not implemented")

    def save_logs_and_checkpoints(
        self, args, global_step, step_bar, logs_dict={}, client_states={}
    ):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update(
                        {
                            f"perf/experience_maker/{k}": v
                            for k, v in self.experience_maker.perf_stats.items()
                        }
                    )
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(
                            f"perf/experience_maker/{k}", v, global_step
                        )

        if global_step % args.eval_steps == 0 or global_step == 1:
            # Changed here such that we evaluate at the first step
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"step_{global_step}"
            # Change here to only save model state dict instead of other states
            if self.strategy.is_rank_0():
                torch.save(
                    self.actor.state_dict(),
                    Path(self.args.save_dir) / f"actor_{tag}.pth",
                )
                if args.save_value_network:
                    torch.save(
                        self.critic.state_dict(),
                        Path(self.args.save_dir) / f"critic_{tag}.pth",
                    )

    def ppo_train(self, global_steps=0):
        if self.trainer_empty_cache:
            torch.cuda.empty_cache()
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=(False if self.strategy.ring_attn_group is not None else True),
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)

        if self.trainer_empty_cache:
            torch.cuda.empty_cache()
        return status_mean

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        """Original training step for actor.

        Add the following changes:
        1. compute the full kl for kl loss
        2. add the logging of gradient norm
        """
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(
                experience.action_log_probs, dim=0
            ).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)],
                dim=0,
            ).unsqueeze(0)
            # pad seq makes the sequence a multiple of ring_attention_size.
            if self.strategy.ring_attn_group is not None:
                (
                    pad_len,
                    sequences,
                    attention_mask,
                    num_actions,
                    packed_seq_lens,
                ) = pad_sequences(
                    sequences,
                    attention_mask,
                    num_actions,
                    packed_seq_lens,
                    self.strategy.ring_attn_group,
                )
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = torch.cat(
                    experience.base_action_log_probs, dim=0
                ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask
            if self.args.use_kl_loss and experience.base_action_log_probs is not None:
                base_action_log_probs = experience.base_action_log_probs

        # actor loss
        action_log_probs, logits = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=self.strategy.ring_attn_group,
            packed_seq_lens=packed_seq_lens,
        )
        # unpad sequence ensures that pad tokens do not contribute to the loss calculation.
        if self.strategy.ring_attn_group is not None:
            assert pad_len is not None
            (
                sequences,
                attention_mask,
                num_actions,
                packed_seq_lens,
                action_log_probs,
                _,
                _,
            ) = unpad_sequences(
                pad_len=pad_len,
                sequences=sequences,
                attention_mask=attention_mask,
                num_actions=num_actions,
                packed_seq_lens=packed_seq_lens,
                action_log_probs=action_log_probs,
                ring_attn_group=self.strategy.ring_attn_group,
            )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            action_mask=experience.action_mask,
        )

        if self.args.use_kl_loss:
            if self.initial_model is not None:
                # change: compute the full kl for kl loss
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    experience.action_mask,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(
                    action_log_probs,
                    dtype=action_log_probs.dtype,
                    device=action_log_probs.device,
                )

            if not self.args.packing_samples:
                kl_mean = masked_mean(kl, experience.action_mask, dim=-1)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.

                kl = unpacking_samples(kl, num_actions)
                kl_mean = torch.tensor(
                    [each_kl.mean() for each_kl in kl],
                    device=action_log_probs.device,
                )

            kl_loss = kl_mean.mean()
            experience.info["kl"] = kl_loss.item()
        else:
            kl_loss = 0

        loss = actor_loss + kl_loss * self.kl_ctl.value

        # entropy calculation
        entropy = compute_entropy(logits)  # (B, T)
        entropy = masked_mean(entropy, experience.action_mask, dim=-1)  # (B,)
        if self.args.entropy_loss_coef != 0:
            entropy_loss = entropy.mean()
            loss -= entropy_loss * self.args.entropy_loss_coef

        # counterpart prediction loss + accuracy
        if self.counterpart_prediction_loss_coef != 0:
            (
                counterpart_prediction_loss,
                counterpart_prediction_acc,
            ) = self.counterpart_prediction_loss_fn(experience, logits)
            loss += counterpart_prediction_loss * self.counterpart_prediction_loss_coef
        else:
            counterpart_prediction_loss = torch.tensor(0.0, device=logits.device)
            counterpart_prediction_acc = torch.tensor(0.0, device=logits.device)

        self.strategy.backward(loss, self.actor, self.actor_optim)
        self.strategy.optimizer_step(
            self.actor_optim, self.actor, self.actor_scheduler, name="actor"
        )

        # status
        status = {
            "policy_loss": actor_loss.item(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
            "entropy": entropy.mean().item(),
            "counterpart_prediction_loss": counterpart_prediction_loss.item(),
            "counterpart_prediction_acc": counterpart_prediction_acc.item(),
        }

        # change: add grad norm to status
        status["grad_norm"] = grad_norm(self.actor, norm_type=2)["grad_2.0_norm_total"]

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum()
                    / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status

    def fit(
        self,
        args,
        train_dataloader,
        eval_dataloader,
    ) -> None:
        # Removed getting eval_steps from steps per episode
        # Removed restore step and start_epoch for now
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        total_steps = args.num_steps
        steps = 1
        self.strategy.print(
            f"Total steps: {total_steps}, "
            f"steps per episode: {self.train_dataloader.__len__()}"
        )

        episode = 0
        pbar = tqdm(
            range(total_steps),
            desc=f"Training",
            disable=not self.strategy.is_rank_0(),
        )
        while True:
            episode += 1
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(episode)
            for batch in self.train_dataloader:
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(
                        batch, is_eval=False, **self.generate_kwargs
                    )
                ):
                    self.replay_buffer.append(experience)

                self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                self.replay_buffer.clear()

                if "kl" in status:
                    self.kl_ctl.update(
                        status["kl"],
                        args.rollout_batch_size * args.n_samples_per_prompt,
                    )
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

                if steps > total_steps:
                    # steps start from 1
                    break

            if steps > total_steps:
                # steps start from 1
                print(f"Training completed at step {steps}")
                break

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()


class SingleMelodyPPOTrainer(ReaLchordsBasePPOTrainer):
    def log_eval(self, experience_to_log, batch_to_log, info_list, global_step):
        if self._tensorboard is not None:
            raise NotImplementedError("Tensorboard logging for eval not implemented")

        # We don't log audio and image into table because there will be bugs
        table_text = self._wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        decoder_preds = experience_to_log.sequences
        targets = batch_to_log["targets"][:, 1:]  # Remove bos token
        for i in range(min(self.max_log_examples, len(targets))):
            decoder_pred = decoder_preds[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            # Convert array to text
            array_pred_text = str(decoder_pred)
            array_gt_text = str(target)
            table_text.add_data(i, array_gt_text, array_pred_text)

            midi_gt = self.output_to_midi(target)
            self.log_midi(midi_gt, suffix=f"gt_{i}")

            try:
                midi_gen = self.output_to_midi(decoder_pred)
                self.log_midi(midi_gen, suffix=f"gen_{i}")

            except Exception as e:
                logger.error(f"Error converting to MIDI: {e}")
                continue

        self._wandb.log({"array_text": table_text})

        song_urls = batch_to_log["song_url"]
        table_url = self._wandb.Table(columns=["Index", "Song URL"])
        for i, song_url in enumerate(song_urls):
            table_url.add_data(i, song_url)

        # Average the info_list
        status_list = info_list[0]
        for m in info_list[1:]:
            for k, v in m.items():
                status_list[k] += v
        for k in status_list.keys():
            status_list[k] /= len(info_list)

        status_mean = {k: v.mean().item() for k, v in status_list.items()}

        # Log to wandb
        if self._wandb is not None and self.strategy.is_rank_0():
            logs = {
                "eval/%s" % k: v
                for k, v in {
                    **status_mean,
                    "global_step": global_step,
                }.items()
            }
            self._wandb.log(logs)

    def output_to_midi(self, decoder_pred):
        midi = self.tokenizer.decode_to_midi(
            melody_frames=decoder_pred,
        )
        return midi

    def log_midi(self, midi, suffix=""):
        audio, image = midi_to_audio_image(midi)
        # Convert audio and image to W&B format
        wandb_audio = self._wandb.Audio(
            audio,
            sample_rate=MIDI_SYNTH_SR,
        )
        wandb_image = self._wandb.Image(image)

        # Log the audio and image to W&B
        self._wandb.log(
            {
                f"audio/{suffix}": wandb_audio,
                f"image/{suffix}": wandb_image,
            }
        )


class ReaLchordsPPOTrainer(ReaLchordsBasePPOTrainer):
    def log_eval(self, experience_to_log, batch_to_log, info_list, global_step):
        if self._tensorboard is not None:
            raise NotImplementedError("Tensorboard logging for eval not implemented")

        # We don't log audio and image into table because there will be bugs
        table_text = self._wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        decoder_preds = experience_to_log.sequences
        decoder_preds = decoder_preds[:, 1:]  # Remove bos token
        targets = batch_to_log["targets"][:, 1:]  # Remove bos token

        note_in_chord_ratio_pred = evaluate_note_in_chord_ratio(
            decoder_preds, self.tokenizer, model_part=self.args.model_part
        )
        note_in_chord_ratio_gt = evaluate_note_in_chord_ratio(
            targets, self.tokenizer, model_part=self.args.model_part
        )

        initial_silence_pred = evaluate_initial_silence(decoder_preds, self.tokenizer)
        initial_silence_gt = evaluate_initial_silence(targets, self.tokenizer)
        average_duration_pred = evaluate_average_duration(decoder_preds, self.tokenizer)
        average_duration_gt = evaluate_average_duration(targets, self.tokenizer)

        for i in range(min(self.max_log_examples, len(targets))):
            decoder_pred = decoder_preds[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            # Convert array to text
            array_pred_text = str(decoder_pred)
            array_gt_text = str(target)
            table_text.add_data(i, array_gt_text, array_pred_text)

            midi_gt = self.output_to_midi(target)
            self.log_midi(midi_gt, suffix=f"gt_{i}")

            try:
                midi_gen = self.output_to_midi(decoder_pred)
                self.log_midi(midi_gen, suffix=f"gen_{i}")

            except Exception as e:
                logger.error(f"Error converting to MIDI: {e}")
                continue

        self._wandb.log({"array_text": table_text})

        song_urls = batch_to_log["song_url"]
        table_url = self._wandb.Table(columns=["Index", "Song URL"])
        for i, song_url in enumerate(song_urls):
            table_url.add_data(i, song_url)

        # Average the info_list
        status_list = info_list[0]
        for m in info_list[1:]:
            for k, v in m.items():
                status_list[k] += v
        for k in status_list.keys():
            status_list[k] /= len(info_list)

        # Add other eval results to status list
        status_list["note_in_chord_ratio_pred"] = note_in_chord_ratio_pred
        status_list["note_in_chord_ratio_gt"] = note_in_chord_ratio_gt
        status_list["avg_initial_silence_pred"] = initial_silence_pred
        status_list["avg_initial_silence_gt"] = initial_silence_gt
        model_part = self.args.model_part
        status_list[f"avg_{model_part}_duration_pred"] = average_duration_pred
        status_list[f"avg_{model_part}_duration_gt"] = average_duration_gt

        status_mean = {k: v.mean().item() for k, v in status_list.items()}

        # Log to wandb
        if self._wandb is not None and self.strategy.is_rank_0():
            logs = {
                "eval/%s" % k: v
                for k, v in {
                    **status_mean,
                    "global_step": global_step,
                }.items()
            }
            self._wandb.log(logs)

    def output_to_midi(self, decoder_pred):
        if self.args.model_part == "melody":
            melody_frames = decoder_pred[::2]
            chord_frames = decoder_pred[1::2]
            midi = self.tokenizer.decode_to_midi(
                melody_frames=melody_frames,
                chord_frames=chord_frames,
            )
        elif self.args.model_part == "chord":
            chord_frames = decoder_pred[::2]
            melody_frames = decoder_pred[1::2]
            midi = self.tokenizer.decode_to_midi(
                chord_frames=chord_frames,
                melody_frames=melody_frames,
            )
        return midi

    def log_midi(self, midi, suffix=""):
        audio, image = midi_to_audio_image(midi)
        # Convert audio and image to W&B format
        wandb_audio = self._wandb.Audio(
            audio,
            sample_rate=MIDI_SYNTH_SR,
        )
        wandb_image = self._wandb.Image(image)

        # Log the audio and image to W&B
        self._wandb.log(
            {
                f"audio/{suffix}": wandb_audio,
                f"image/{suffix}": wandb_image,
            }
        )


class GAILMixin:
    """Mixin class that adds GAIL (Generative Adversarial Imitation Learning) functionality to PPO trainers."""

    def __init__(self, *args, **kwargs):
        self.reward_optim = kwargs.pop("reward_optim", None)
        self.reward_scheduler = kwargs.pop("reward_scheduler", None)
        self.reward_fn_name = kwargs.pop("reward_fn_name", None)
        self.reward_update_steps = kwargs.pop("reward_update_steps", 1)
        self.reward_update_early_stop_steps = kwargs.pop(
            "reward_update_early_stop_steps", 0
        )
        self.enable_reward_label_smoothing = kwargs.pop(
            "enable_reward_label_smoothing", False
        )
        if self.enable_reward_label_smoothing:
            label_smoothing = 0.1
        else:
            label_smoothing = 0
        self.reward_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        assert self.reward_optim is not None
        assert self.reward_scheduler is not None
        self.reward_update_strategy = kwargs.pop("reward_update_strategy", "steps")
        self.reward_average_steps = kwargs.pop("reward_average_steps", None)
        self.reward_update_threshold = kwargs.pop("reward_update_threshold", None)
        self.reward_apply_threshold_after_steps = kwargs.pop(
            "reward_apply_threshold_after_steps", None
        )
        if self.reward_update_strategy == "average":
            assert self.reward_average_steps is not None
            assert self.reward_update_threshold is not None
            assert self.reward_apply_threshold_after_steps is not None
        self.reward_history = []
        # Because for each global step (i.e. one episode), we will train multiple steps
        # We need to store the reward of each step
        self.reward_current_step = []
        self.current_step = None
        super().__init__(*args, **kwargs)

    def train_reward_this_step(self, global_steps):
        if self.reward_update_strategy == "steps":
            if global_steps % self.reward_update_steps == 0 or global_steps == 1:
                if (
                    self.reward_update_early_stop_steps is not None
                    and global_steps > self.reward_update_early_stop_steps
                ):
                    return False  # early stop
                return True
            return False
        elif self.reward_update_strategy == "average":
            recent_rewards = self.reward_history[-self.reward_average_steps :]
            above_threshold = np.mean(recent_rewards) > self.reward_update_threshold
            before_threshold_steps = (
                global_steps < self.reward_apply_threshold_after_steps
            )
            is_update_step = (
                global_steps % self.reward_update_steps == 0 or global_steps == 1
            )
            if above_threshold or (before_threshold_steps and is_update_step):
                return True
            else:
                return False
        else:
            raise ValueError(
                f"Invalid reward update strategy: {self.reward_update_strategy}"
            )

    def add_to_reward_history(self, experience: Experience, global_steps: int):
        if self.current_step is not None and self.current_step != global_steps:
            self.reward_history.append(np.mean(self.reward_current_step))
            self.reward_current_step = []
        self.current_step = global_steps
        self.reward_current_step.append(
            experience.info["reward_gail_reward"].mean().item()
        )

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        self.add_to_reward_history(experience, global_steps)
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience)
        if self.critic is not None:
            status.update(self.training_step_critic(experience))
        if self.reward_optim is not None:
            if self.train_reward_this_step(global_steps):
                print(f"Training reward at step {global_steps}")
                status.update(self.training_step_reward(experience))
        return status

    def training_step_reward(self, experience: Experience) -> Dict[str, float]:
        reward_fn_idx = self.experience_maker.reward_names.index(self.reward_fn_name)
        reward_fn = self.experience_maker.reward_fns[reward_fn_idx]
        reward_model = reward_fn.model

        # Handle VRAM swap: move model to correct device if needed
        if self.experience_maker.reward_vram_swap and isinstance(reward_fn, nn.Module):
            original_device = self.experience_maker.reward_device_dict[
                self.reward_fn_name
            ]
            reward_model.to(original_device)

        reward_model.train()

        sequences = experience.sequences
        targets = experience.targets
        targets = targets.to(sequences.device)
        targets = targets[:, :-1]  # remove eos token

        tokens_gen, mask_gen = reward_fn.get_inputs_from_sequence(sequences)
        tokens_data, mask_data = reward_fn.get_inputs_from_sequence(targets)
        tokens = torch.cat([tokens_gen, tokens_data], dim=0)
        mask = torch.cat([mask_gen, mask_data], dim=0)

        # GAN label: 0 for generated, 1 for data
        labels = torch.cat(
            [
                torch.zeros(tokens_gen.size(0), dtype=torch.long),
                torch.ones(tokens_data.size(0), dtype=torch.long),
            ],
            dim=0,
        )
        labels = labels.to(sequences.device)

        # reward loss
        logits = reward_model(tokens, mask)

        # loss function
        reward_loss = self.reward_loss_fn(logits, labels)

        loss = reward_loss
        self.strategy.backward(loss, reward_model, self.reward_optim)
        self.strategy.optimizer_step(
            self.reward_optim,
            reward_model,
            self.reward_scheduler,
            name="reward",
        )

        # reward accuracy
        reward_acc = (logits.argmax(dim=-1) == labels).float().mean()

        # status
        status = {
            "reward_loss": reward_loss.item(),
            "reward_lr": self.reward_scheduler.get_last_lr()[0],
            "reward_acc": reward_acc.item(),
        }

        reward_model.eval()

        # Handle VRAM swap: move model back to CPU to save memory
        if self.experience_maker.reward_vram_swap and isinstance(reward_fn, nn.Module):
            reward_model.cpu()

        return status


class ReaLchordsGAILPPOTrainer(GAILMixin, ReaLchordsPPOTrainer):
    """ReaLchords PPO trainer with GAIL functionality."""

    pass
