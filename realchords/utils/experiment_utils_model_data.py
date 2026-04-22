"""Utilities for model-vs-data sequence generation experiments."""

from typing import Any, List, Optional, Tuple

import torch
from tqdm import tqdm

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.experiment_utils import (
    compute_social_influence_kl,
    get_online_and_semi_online_model,
    replace_eos_with_pad,
)
from realchords.utils.experiment_utils_data_perturbation import (
    apply_data_perturbation,
)
from realchords.utils.sequence_utils import sequences_order_to_counterpart

MAX_GEN_STEPS = 512


def extract_prompts_from_data(
    sequences: torch.Tensor, prompt_steps: int, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Extract BOS-prefixed prompts from interleaved data sequences."""
    if prompt_steps == 0:
        return torch.full(
            (sequences.shape[0], 1),
            tokenizer.bos_token,
            dtype=torch.long,
            device=sequences.device,
        )

    prompt_tokens = sequences[:, : prompt_steps * 2]
    bos_tokens = torch.full(
        (sequences.shape[0], 1),
        tokenizer.bos_token,
        dtype=torch.long,
        device=sequences.device,
    )

    return torch.cat([bos_tokens, prompt_tokens], dim=1)


def adjust_conditions_for_prompts(
    sequences: torch.Tensor,
    prompt_steps: int,
) -> torch.Tensor:
    """Adjust conditions to account for the prompted portion."""
    if prompt_steps == 0:
        return sequences[:, 1::2]

    start_idx = prompt_steps * 2
    remaining_seq = sequences[:, start_idx:]
    return remaining_seq[:, 1::2]


def convert_sequence_order_for_model(
    sequences: torch.Tensor, target_model_type: str, source_model_type: str
) -> torch.Tensor:
    """Convert sequence order between chord-order and melody-order."""
    if target_model_type == source_model_type:
        return sequences
    return sequences_order_to_counterpart(sequences)


def generate_from_data(
    model: torch.nn.Module,
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    prompt_steps: int = 0,
    target_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Generate sequences from data conditions and return prompt + generated tokens."""
    if target_seq_len is None:
        target_seq_len = MAX_GEN_STEPS

    if target_seq_len % 2 == 0:
        target_seq_len += 1

    prompts = extract_prompts_from_data(sequences, prompt_steps, tokenizer)
    prompt_len = prompts.shape[1]
    tokens_to_generate = target_seq_len - prompt_len

    if tokens_to_generate <= 0:
        raise ValueError(
            f"Invalid configuration: target_seq_len ({target_seq_len}) must be greater than prompt length ({prompt_len})."
        )
    if tokens_to_generate % 2 != 0:
        raise ValueError(
            f"Internal error: tokens_to_generate ({tokens_to_generate}) should be even."
        )

    conditions = adjust_conditions_for_prompts(sequences, prompt_steps)

    full_conditions = sequences[:, 1::2]
    conditions_mask = full_conditions != tokenizer.pad_token

    output_mask = torch.zeros(
        (conditions.shape[0], target_seq_len - 1),
        dtype=torch.bool,
        device=sequences.device,
    )
    output_mask[:, 0::2][:, : conditions_mask.shape[1]] = conditions_mask
    output_mask[:, 1::2][:, : conditions_mask.shape[1]] = conditions_mask
    output_mask = torch.cat(
        [
            torch.ones(
                (conditions.shape[0], 1),
                dtype=torch.bool,
                device=sequences.device,
            ),
            output_mask,
        ],
        dim=1,
    )

    with torch.no_grad():
        generated_part = model.generate_online(
            prompts,
            conditions=conditions,
            seq_len=tokens_to_generate,
            cache_kv=True,
        )
        complete_sequence = torch.cat([prompts, generated_part], dim=1)
        complete_sequence.masked_fill_(~output_mask, tokenizer.pad_token)

    return complete_sequence


def handle_data_conditioned_generation(
    args: Any,
    mel_source: str,
    chord_source: str,
    melody_model: Optional[torch.nn.Module],
    chord_model: Optional[torch.nn.Module],
    melody_tokenizer: HooktheoryTokenizer,
    chord_tokenizer: HooktheoryTokenizer,
    melody_dataloaders: Tuple,
    chord_dataloaders: Tuple,
    mle_melody_model: torch.nn.Module,
    mle_chord_model: torch.nn.Module,
    device: torch.device,
    data_perturbation: str = "none",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Handle data-conditioned generation (e.g. MLE chord conditioned on melody data)."""
    generated_all = []

    condition_type = "melody" if mel_source == "melody_data" else "chord"
    model_part = "melody" if condition_type == "chord" else "chord"
    model_to_use = chord_model if mel_source == "melody_data" else melody_model
    tokenizer_to_use = (
        chord_tokenizer if mel_source == "melody_data" else melody_tokenizer
    )
    dataloaders_to_use = (
        melody_dataloaders if model_part == "melody" else chord_dataloaders
    )

    _, val_dataloader = dataloaders_to_use
    print(f"Validation dataloader has {len(val_dataloader)} batches.")

    if args.num_batches == -1:
        num_batches_to_process = len(val_dataloader)
        print(
            f"Processing all {num_batches_to_process} batches from validation dataloader"
        )
    elif args.num_batches > len(val_dataloader):
        raise ValueError(
            f"num_batches ({args.num_batches}) cannot be greater than validation dataloader length ({len(val_dataloader)})"
        )
    else:
        num_batches_to_process = args.num_batches
        print(
            f"Processing {num_batches_to_process} batches from validation dataloader"
        )

    print(
        f"Generating {chord_source if mel_source == 'melody_data' else mel_source} conditioned on {condition_type} data"
    )

    if args.prompt_steps > 0:
        print(f"Using {args.prompt_steps} frames as prompts from data")

    for batch_idx, batch in enumerate(
        tqdm(val_dataloader, total=num_batches_to_process)
    ):
        if batch_idx >= num_batches_to_process:
            break

        sequences = batch["targets"].to(device)[:, 1:-1]
        sequences = replace_eos_with_pad(sequences, tokenizer_to_use)

        if data_perturbation != "none":
            sequences = apply_data_perturbation(
                sequences, data_perturbation, condition_type, tokenizer_to_use
            )

        decoder_preds = generate_from_data(
            model_to_use,
            sequences,
            tokenizer_to_use,
            prompt_steps=args.prompt_steps,
            target_seq_len=args.target_seq_len,
        )

        assert (
            decoder_preds[0, 0] == tokenizer_to_use.bos_token
        ), "BOS token should be at the beginning"

        online_model, semi_online_model = get_online_and_semi_online_model(
            model_part,
            mle_melody_model,
            mle_chord_model,
            melody_model,
            chord_model,
        )
        kl = compute_social_influence_kl(
            decoder_preds, online_model, semi_online_model
        )

        generated_all.append((decoder_preds.cpu(), kl.cpu()))

    return generated_all
