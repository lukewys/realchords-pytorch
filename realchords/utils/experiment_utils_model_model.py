"""Utilities for model-vs-model and agent-switching sequence generation."""

from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.lit_module.decoder_only import LitDecoder
from realchords.model.sampling import filter_special_token
from realchords.rl.marl_interaction import generate_marl
from realchords.utils.experiment_utils import (
    compute_social_influence_kl,
    create_dataset_dataloaders,
    ensure_bos_token,
    get_online_and_semi_online_model,
    replace_eos_with_pad,
)
from realchords.utils.experiment_utils_model_data import (
    convert_sequence_order_for_model,
    extract_prompts_from_data,
)
from realchords.utils.inference_utils import load_lit_model, load_rl_model


def generate_from_marl(
    chord_model: torch.nn.Module,
    melody_model: torch.nn.Module,
    chord_tokenizer: HooktheoryTokenizer,
    melody_tokenizer: HooktheoryTokenizer,
    gen_inputs: torch.Tensor,
    target_seq_len: int,
    prompt_data: Optional[torch.Tensor] = None,
    prompt_steps: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequences using MARL and return prompt + generated tokens."""
    if target_seq_len % 2 == 0:
        target_seq_len += 1
        print(
            f"Adjusted target_seq_len to {target_seq_len} to ensure odd total length"
        )

    with torch.no_grad():
        if prompt_steps > 0 and prompt_data is not None:
            prompts = extract_prompts_from_data(
                prompt_data, prompt_steps, chord_tokenizer
            )
            prompt_len = prompts.shape[1]
            tokens_to_generate = target_seq_len - prompt_len

            if tokens_to_generate <= 0:
                raise ValueError(
                    f"Invalid configuration: target_seq_len ({target_seq_len}) must be greater than prompt length ({prompt_len})."
                )

            generated_seq_1, generated_seq_2 = generate_marl(
                chord_model,
                melody_model,
                prompts,
                tokens_to_generate,
                cache_kv=True,
                debug=False,
                filter_logits_fn_1=filter_special_token,
                filter_logits_fn_2=filter_special_token,
            )

            complete_seq_1 = torch.cat([prompts, generated_seq_1], dim=1)
            prompts_melody_order = convert_sequence_order_for_model(
                prompts, "melody", "chord"
            )
            complete_seq_2 = torch.cat(
                [prompts_melody_order, generated_seq_2], dim=1
            )
        else:
            tokens_to_generate = target_seq_len - 1
            generated_seq_1, generated_seq_2 = generate_marl(
                chord_model,
                melody_model,
                gen_inputs,
                tokens_to_generate,
                cache_kv=True,
                debug=False,
                filter_logits_fn_1=filter_special_token,
                filter_logits_fn_2=filter_special_token,
            )
            complete_seq_1 = generated_seq_1
            complete_seq_2 = generated_seq_2

    return complete_seq_1, complete_seq_2


def generate_from_marl_with_switching(
    chord_model_or_models: Union[torch.nn.Module, List[torch.nn.Module]],
    melody_model_or_models: Union[torch.nn.Module, List[torch.nn.Module]],
    chord_tokenizer: HooktheoryTokenizer,
    melody_tokenizer: HooktheoryTokenizer,
    agent_switch_steps: List[int],
    target_seq_len: int,
    switching_type: str,
    prompt_data: Optional[torch.Tensor] = None,
    prompt_steps: int = 0,
    batch_size: int = 64,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sequences using MARL with agent switching."""
    if target_seq_len % 2 == 0:
        target_seq_len += 1

    if switching_type == "melody":
        chord_model = chord_model_or_models
        melody_models = melody_model_or_models
    else:
        chord_models = chord_model_or_models
        melody_model = melody_model_or_models

    gen_inputs = torch.full(
        (batch_size, 1),
        chord_tokenizer.bos_token,
        dtype=torch.long,
        device=device,
    )

    if prompt_steps > 0 and prompt_data is not None:
        prompt_tokens = 1 + prompt_steps * 2
    else:
        prompt_tokens = 1

    switch_tokens = sum(agent_switch_steps) * 2
    expected_total = prompt_tokens + switch_tokens

    if expected_total != target_seq_len:
        raise ValueError(
            f"Token count mismatch: prompt_tokens ({prompt_tokens}) + switch_tokens ({switch_tokens}) = {expected_total}, but target_seq_len is {target_seq_len}."
        )

    if prompt_steps > 0 and prompt_data is not None:
        prompts = extract_prompts_from_data(
            prompt_data, prompt_steps, chord_tokenizer
        )
        current_seq_1 = prompts.clone()
        current_seq_2 = convert_sequence_order_for_model(
            prompts, "melody", "chord"
        ).clone()
    else:
        current_seq_1 = gen_inputs.clone()
        current_seq_2 = gen_inputs.clone()

    print(
        f"Using switch frames: {agent_switch_steps} (total {switch_tokens} tokens)"
    )

    with torch.no_grad():
        for agent_idx, steps_for_agent in enumerate(agent_switch_steps):
            if switching_type == "melody":
                current_chord_model = chord_model
                current_melody_model = melody_models[
                    agent_idx % len(melody_models)
                ]
            else:
                current_chord_model = chord_models[
                    agent_idx % len(chord_models)
                ]
                current_melody_model = melody_model

            tokens_to_generate = steps_for_agent * 2

            print(
                f"Agent {agent_idx}: generating {tokens_to_generate} tokens with {switching_type} agent {agent_idx % len(melody_models if switching_type == 'melody' else chord_models)}"
            )

            if tokens_to_generate > 0:
                generated_seq_1, generated_seq_2 = generate_marl(
                    current_chord_model,
                    current_melody_model,
                    current_seq_1,
                    tokens_to_generate,
                    cache_kv=True,
                    debug=False,
                    filter_logits_fn_1=filter_special_token,
                    filter_logits_fn_2=filter_special_token,
                )

                current_seq_1 = torch.cat(
                    [current_seq_1, generated_seq_1], dim=1
                )
                current_seq_2 = torch.cat(
                    [current_seq_2, generated_seq_2], dim=1
                )

    return current_seq_1, current_seq_2


def load_models_for_switching(
    args: Any, switching_type: str, fixed_source: str, device: torch.device
) -> Tuple[
    List[torch.nn.Module],
    torch.nn.Module,
    HooktheoryTokenizer,
    HooktheoryTokenizer,
    Tuple,
    Tuple,
    torch.nn.Module,
    torch.nn.Module,
]:
    """Load models for agent switching modes."""
    mle_melody_model, melody_tokenizer, _ = load_lit_model(
        model_path=args.mle_melody_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.batch_size,
        compile=False,
    )

    mle_chord_model, chord_tokenizer, _ = load_lit_model(
        model_path=args.mle_chord_model_path,
        lit_module_cls=LitDecoder,
        batch_size=args.batch_size,
        compile=False,
    )

    melody_dataloaders = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="melody",
        batch_size=args.batch_size,
        max_len=args.target_seq_len,
    )
    chord_dataloaders = create_dataset_dataloaders(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        model_part="chord",
        batch_size=args.batch_size,
        max_len=args.target_seq_len,
    )

    switching_models = []
    fixed_model = None

    if switching_type == "melody":
        model_paths = args.rl_melody_model_paths or args.mle_melody_model_paths
        use_rl = args.rl_melody_model_paths is not None

        for path in model_paths:
            if use_rl:
                model = load_rl_model(
                    model_path=path,
                    model=deepcopy(mle_melody_model),
                    compile=False,
                )
            else:
                model, _, _ = load_lit_model(
                    model_path=path,
                    lit_module_cls=LitDecoder,
                    batch_size=args.batch_size,
                    compile=False,
                )
            model.eval().to(device)
            switching_models.append(model)

        if fixed_source == "rl_chord":
            fixed_model = load_rl_model(
                model_path=args.rl_chord_model_path,
                model=deepcopy(mle_chord_model),
                compile=False,
            )
        else:
            fixed_model = mle_chord_model
        fixed_model.eval().to(device)

    elif switching_type == "chord":
        model_paths = args.rl_chord_model_paths or args.mle_chord_model_paths
        use_rl = args.rl_chord_model_paths is not None

        for path in model_paths:
            if use_rl:
                model = load_rl_model(
                    model_path=path,
                    model=deepcopy(mle_chord_model),
                    compile=False,
                )
            else:
                model, _, _ = load_lit_model(
                    model_path=path,
                    lit_module_cls=LitDecoder,
                    batch_size=args.batch_size,
                    compile=False,
                )
            model.eval().to(device)
            switching_models.append(model)

        if fixed_source == "rl_melody":
            fixed_model = load_rl_model(
                model_path=args.rl_melody_model_path,
                model=deepcopy(mle_melody_model),
                compile=False,
            )
        else:
            fixed_model = mle_melody_model
        fixed_model.eval().to(device)
    else:
        raise ValueError(f"Invalid switching_type: {switching_type}")
    mle_melody_model.eval().to(device)
    mle_chord_model.eval().to(device)

    if melody_tokenizer is None:
        melody_tokenizer = chord_tokenizer
    if chord_tokenizer is None:
        chord_tokenizer = melody_tokenizer

    return (
        switching_models,
        fixed_model,
        melody_tokenizer,
        chord_tokenizer,
        melody_dataloaders,
        chord_dataloaders,
        mle_melody_model,
        mle_chord_model,
    )


def handle_model_vs_model_generation(
    args: Any,
    chord_model: torch.nn.Module,
    melody_model: torch.nn.Module,
    chord_tokenizer: HooktheoryTokenizer,
    melody_tokenizer: HooktheoryTokenizer,
    chord_dataloaders: Tuple,
    mle_melody_model: torch.nn.Module,
    mle_chord_model: torch.nn.Module,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Handle model-vs-model generation (e.g. RL melody vs RL chord)."""
    generated_all = []

    print("Generating with model vs model")

    prompt_data_iter = None
    if args.prompt_steps > 0:
        _, val_dataloader = chord_dataloaders
        prompt_data_iter = iter(val_dataloader)
        print(f"Using {args.prompt_steps} frames as prompts from data")

    for _ in tqdm(range(args.num_batches)):
        prompt_data = None
        if args.prompt_steps > 0:
            try:
                batch = next(prompt_data_iter)
            except StopIteration:
                prompt_data_iter = iter(val_dataloader)
                batch = next(prompt_data_iter)
            prompt_data = batch["targets"].to(device)[: args.batch_size, 1:-1]
            prompt_data = replace_eos_with_pad(prompt_data, chord_tokenizer)

        gen_inputs = torch.full(
            (args.batch_size, 1),
            chord_tokenizer.bos_token,
            dtype=torch.long,
            device=device,
        )

        generated_seq_1, generated_seq_2 = generate_from_marl(
            chord_model,
            melody_model,
            chord_tokenizer,
            melody_tokenizer,
            gen_inputs,
            args.target_seq_len,
            prompt_data=prompt_data,
            prompt_steps=args.prompt_steps,
        )

        generated_seq_1 = ensure_bos_token(generated_seq_1, chord_tokenizer)
        generated_seq_2 = ensure_bos_token(generated_seq_2, melody_tokenizer)

        online_model, semi_online_model = get_online_and_semi_online_model(
            "chord",
            mle_melody_model,
            mle_chord_model,
            melody_model,
            chord_model,
        )
        kl_chord = compute_social_influence_kl(
            generated_seq_1, online_model, semi_online_model
        )

        online_model, semi_online_model = get_online_and_semi_online_model(
            "melody",
            mle_melody_model,
            mle_chord_model,
            melody_model,
            chord_model,
        )
        kl_melody = compute_social_influence_kl(
            generated_seq_2, online_model, semi_online_model
        )

        generated_all.append(
            (
                generated_seq_1.cpu(),
                generated_seq_2.cpu(),
                kl_chord.cpu(),
                kl_melody.cpu(),
            )
        )

    return generated_all


def handle_agent_switching_generation(
    args: Any,
    switching_models: List[torch.nn.Module],
    fixed_model: torch.nn.Module,
    melody_tokenizer: HooktheoryTokenizer,
    chord_tokenizer: HooktheoryTokenizer,
    chord_dataloaders: Tuple,
    mle_melody_model: torch.nn.Module,
    mle_chord_model: torch.nn.Module,
    switching_type: str,
    fixed_source: str,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Handle agent switching generation."""
    generated_all = []
    switching_melody = switching_type == "melody"

    print(f"Generating with agent switching: {switching_type} agents switching")
    print(f"Fixed agent type: {fixed_source}")
    print(f"Number of switching agents: {len(switching_models)}")
    print(f"Switch frames: {args.agent_switch_steps}")

    prompt_data_iter = None
    if args.prompt_steps > 0:
        _, val_dataloader = chord_dataloaders
        prompt_data_iter = iter(val_dataloader)
        print(f"Using {args.prompt_steps} frames as prompts from data")

    for _ in tqdm(range(args.num_batches)):
        prompt_data = None
        if args.prompt_steps > 0:
            try:
                batch = next(prompt_data_iter)
            except StopIteration:
                prompt_data_iter = iter(val_dataloader)
                batch = next(prompt_data_iter)
            prompt_data = batch["targets"].to(device)[: args.batch_size, 1:-1]
            prompt_data = replace_eos_with_pad(prompt_data, chord_tokenizer)

        if switching_melody:
            generated_seq_1, generated_seq_2 = (
                generate_from_marl_with_switching(
                    fixed_model,
                    switching_models,
                    chord_tokenizer,
                    melody_tokenizer,
                    args.agent_switch_steps,
                    args.target_seq_len,
                    switching_type="melody",
                    prompt_data=prompt_data,
                    prompt_steps=args.prompt_steps,
                    batch_size=args.batch_size,
                    device=device,
                )
            )
        else:
            generated_seq_1, generated_seq_2 = (
                generate_from_marl_with_switching(
                    switching_models,
                    fixed_model,
                    chord_tokenizer,
                    melody_tokenizer,
                    args.agent_switch_steps,
                    args.target_seq_len,
                    switching_type="chord",
                    prompt_data=prompt_data,
                    prompt_steps=args.prompt_steps,
                    batch_size=args.batch_size,
                    device=device,
                )
            )

        generated_seq_1 = ensure_bos_token(generated_seq_1, chord_tokenizer)
        generated_seq_2 = ensure_bos_token(generated_seq_2, melody_tokenizer)

        if switching_melody:
            online_model, semi_online_model = get_online_and_semi_online_model(
                "chord",
                mle_melody_model,
                mle_chord_model,
                switching_models[-1],
                fixed_model,
            )
            kl_chord = compute_social_influence_kl(
                generated_seq_1, online_model, semi_online_model
            )

            online_model, semi_online_model = get_online_and_semi_online_model(
                "melody",
                mle_melody_model,
                mle_chord_model,
                switching_models[-1],
                fixed_model,
            )
            kl_melody = compute_social_influence_kl(
                generated_seq_2, online_model, semi_online_model
            )
        else:
            online_model, semi_online_model = get_online_and_semi_online_model(
                "chord",
                mle_melody_model,
                mle_chord_model,
                fixed_model,
                switching_models[-1],
            )
            kl_chord = compute_social_influence_kl(
                generated_seq_1, online_model, semi_online_model
            )

            online_model, semi_online_model = get_online_and_semi_online_model(
                "melody",
                mle_melody_model,
                mle_chord_model,
                fixed_model,
                switching_models[-1],
            )
            kl_melody = compute_social_influence_kl(
                generated_seq_2, online_model, semi_online_model
            )

        generated_all.append(
            (
                generated_seq_1.cpu(),
                generated_seq_2.cpu(),
                kl_chord.cpu(),
                kl_melody.cpu(),
            )
        )

    return generated_all
