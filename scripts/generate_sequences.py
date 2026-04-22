#!/usr/bin/env python3
"""Generate evaluation sequences from trained checkpoints.

This CLI supports three generation regimes:

1. Model-vs-model MARL generation
   Example:
       python scripts/generate_sequences.py \
           --mode rl_melody_vs_rl_chord \
           --rl_melody_model_path logs/melody_rl/actor.pth \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --save_dir logs/generated/rl_vs_rl \
           --num_batches 16

2. Data-conditioned generation with optional perturbation
   Example:
       python scripts/generate_sequences.py \
           --mode melody_data_vs_rl_chord \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --dataset_name wikifonia \
           --dataset_split test \
           --data_perturbation multiple_transpose \
           --save_dir logs/generated/ood \
           --num_batches -1

3. Agent-switching generation
   Example:
       python scripts/generate_sequences.py \
           --mode rl_chord_vs_switching_melody \
           --rl_chord_model_path logs/chord_rl/actor.pth \
           --rl_melody_model_paths logs/melody_a/actor.pth logs/melody_b/actor.pth \
           --agent_switch_frames 64 64 \
           --target_seq_len 257 \
           --save_dir logs/generated/switching \
           --num_batches 8

Generated artifacts:
  - model-vs-model and agent switching modes:
      <mode>_generated_chord_order.pt
      <mode>_generated_melody_order.pt
      <mode>_kl_chord.pt
      <mode>_kl_melody.pt
  - data-conditioned modes:
      <mode>_generated.pt
      <mode>_kl.pt
  - data-only mode:
      <mode>_chord_order.pt

All sequence tensors are rank-2 integer tensors with one sequence per row.
Outputs intended for downstream evaluation keep BOS at column 0 when generation
produces it, matching the private evaluation pipeline.
"""

import argparse

import torch
from lightning import seed_everything

from realchords.utils.experiment_utils import (
    handle_data_only_mode,
    load_models,
    save_generated_sequences,
)
from realchords.utils.experiment_utils_data_perturbation import (
    validate_perturbation_args,
)
from realchords.utils.experiment_utils_model_data import (
    MAX_GEN_STEPS,
    handle_data_conditioned_generation,
)
from realchords.utils.experiment_utils_model_model import (
    handle_agent_switching_generation,
    handle_model_vs_model_generation,
    load_models_for_switching,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sequences for evaluation from MLE/RL checkpoints."
    )
    parser.add_argument(
        "--rl_melody_model_path",
        type=str,
        default=None,
        help="Path to the RL melody checkpoint (.pth).",
    )
    parser.add_argument(
        "--rl_chord_model_path",
        type=str,
        default=None,
        help="Path to the RL chord checkpoint (.pth).",
    )
    parser.add_argument(
        "--mle_melody_model_path",
        type=str,
        default="logs/decoder_only_online_melody/step=11000.ckpt",
        help="Path to the baseline MLE melody Lightning checkpoint.",
    )
    parser.add_argument(
        "--mle_chord_model_path",
        type=str,
        default="logs/decoder_only_online_chord/step=11000.ckpt",
        help="Path to the baseline MLE chord Lightning checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Generation mode, e.g. rl_melody_vs_rl_chord or melody_data_vs_rl_chord.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where generated tensors and KL artifacts will be written.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help=(
            "Number of batches to process. Use -1 only for data-conditioned modes to process the full validation split."
        ),
    )
    parser.add_argument(
        "--target_seq_len",
        type=int,
        default=MAX_GEN_STEPS,
        help="Target length of the final sequence in tokens, including prompts.",
    )
    parser.add_argument(
        "--prompt_frames",
        dest="prompt_steps",
        type=int,
        default=0,
        help="Number of data frames used as prompts before free generation.",
    )
    parser.add_argument(
        "--rl_melody_model_paths",
        type=str,
        nargs="*",
        default=None,
        help="List of RL melody checkpoints for agent switching.",
    )
    parser.add_argument(
        "--mle_melody_model_paths",
        type=str,
        nargs="*",
        default=None,
        help="List of MLE melody checkpoints for agent switching.",
    )
    parser.add_argument(
        "--rl_chord_model_paths",
        type=str,
        nargs="*",
        default=None,
        help="List of RL chord checkpoints for agent switching.",
    )
    parser.add_argument(
        "--mle_chord_model_paths",
        type=str,
        nargs="*",
        default=None,
        help="List of MLE chord checkpoints for agent switching.",
    )
    parser.add_argument(
        "--agent_switch_frames",
        dest="agent_switch_steps",
        type=int,
        nargs="*",
        default=None,
        help="Frame counts for each switching segment. Each frame corresponds to one chord+melody pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--data_perturbation",
        type=str,
        choices=["none", "multiple_transpose", "single_transpose_6"],
        default="none",
        help="Optional perturbation applied to conditioning data in model-vs-data modes.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["hooktheory", "pop909", "nottingham", "wikifonia"],
        default="hooktheory",
        help="Dataset from which conditioning examples are drawn in data modes.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        choices=["train", "valid", "test"],
        default="valid",
        help="Dataset split used for conditioning examples.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace, mode_parts: list[str]) -> None:
    is_data_mode = "data" in mode_parts[0] or "data" in mode_parts[1]
    if not is_data_mode and args.num_batches < 1:
        raise ValueError(
            "num_batches must be >= 1 for model-vs-model and switching modes. Use -1 only with data-conditioned generation."
        )


def main() -> None:
    args = get_args()
    seed_everything(args.seed)

    mode_parts = args.mode.split("_vs_")
    if len(mode_parts) != 2:
        raise ValueError(
            "Mode should be in the format 'mel_source_vs_chord_source'"
        )

    validate_args(args, mode_parts)

    is_switching_mode = mode_parts[0] in [
        "switching_melody",
        "switching_chord",
    ] or mode_parts[1] in ["switching_melody", "switching_chord"]

    if is_switching_mode:
        if args.agent_switch_steps is None:
            raise ValueError(
                "agent_switch_steps must be provided for switching modes"
            )

        if (
            mode_parts[0] == "switching_melody"
            or mode_parts[1] == "switching_melody"
        ):
            if (
                not args.rl_melody_model_paths
                and not args.mle_melody_model_paths
            ):
                raise ValueError(
                    "rl_melody_model_paths or mle_melody_model_paths must be provided for switching_melody mode"
                )
        if (
            mode_parts[0] == "switching_chord"
            or mode_parts[1] == "switching_chord"
        ):
            if not args.rl_chord_model_paths and not args.mle_chord_model_paths:
                raise ValueError(
                    "rl_chord_model_paths or mle_chord_model_paths must be provided for switching_chord mode"
                )

    valid_sources = [
        "rl_melody",
        "mle_melody",
        "melody_data",
        "switching_melody",
        "rl_chord",
        "mle_chord",
        "switching_chord",
    ]
    valid_targets = [
        "rl_chord",
        "mle_chord",
        "chord_data",
        "switching_chord",
        "rl_melody",
        "mle_melody",
        "switching_melody",
    ]

    if mode_parts[0] not in valid_sources or mode_parts[1] not in valid_targets:
        raise ValueError(
            "Invalid mode. See --help for supported mode families."
        )

    validate_perturbation_args(args.mode, args.data_perturbation)

    mel_source, chord_source = mode_parts[0], mode_parts[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "switching_melody" in mode_parts:
        switching_type = "melody"
        fixed_source = (
            mode_parts[1]
            if mode_parts[0] == "switching_melody"
            else mode_parts[0]
        )
    elif "switching_chord" in mode_parts:
        switching_type = "chord"
        fixed_source = (
            mode_parts[1]
            if mode_parts[0] == "switching_chord"
            else mode_parts[0]
        )
    else:
        switching_type = None
        fixed_source = None

    print(f"Mode: {args.mode}")
    print(f"Switching mode: {switching_type is not None}")
    print(f"Mel source: {mel_source}")
    print(f"Chord source: {chord_source}")

    if switching_type is not None:
        (
            switching_models,
            fixed_model,
            melody_tokenizer,
            chord_tokenizer,
            melody_dataloaders,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
        ) = load_models_for_switching(
            args, switching_type, fixed_source, device
        )

        generated_all = handle_agent_switching_generation(
            args,
            switching_models,
            fixed_model,
            melody_tokenizer,
            chord_tokenizer,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
            switching_type,
            fixed_source,
            device,
        )
    else:
        (
            melody_model,
            chord_model,
            melody_tokenizer,
            chord_tokenizer,
            melody_dataloaders,
            chord_dataloaders,
            mle_melody_model,
            mle_chord_model,
        ) = load_models(args, mel_source, chord_source, device)

        if mel_source == "melody_data" and chord_source == "chord_data":
            handle_data_only_mode(
                args, chord_dataloaders, device, args.data_perturbation
            )
            return
        if "data" not in mel_source and "data" not in chord_source:
            generated_all = handle_model_vs_model_generation(
                args,
                chord_model,
                melody_model,
                chord_tokenizer,
                melody_tokenizer,
                chord_dataloaders,
                mle_melody_model,
                mle_chord_model,
                device,
            )
        else:
            generated_all = handle_data_conditioned_generation(
                args,
                mel_source,
                chord_source,
                melody_model,
                chord_model,
                melody_tokenizer,
                chord_tokenizer,
                melody_dataloaders,
                chord_dataloaders,
                mle_melody_model,
                mle_chord_model,
                device,
                data_perturbation=args.data_perturbation,
            )

    if switching_type is not None:
        if switching_type == "melody":
            save_generated_sequences(
                generated_all, args, "switching_melody", fixed_source
            )
        else:
            save_generated_sequences(
                generated_all, args, fixed_source, "switching_chord"
            )
    else:
        save_generated_sequences(generated_all, args, mel_source, chord_source)


if __name__ == "__main__":
    main()
