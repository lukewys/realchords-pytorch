#!/usr/bin/env python3
"""Analyze chord diversity statistics for generated sequences.

For each generated sequence tensor produced by `scripts/generate_sequences.py`,
this script computes two diversity views:

1. Frame-level chord entropy from the empirical distribution of decoded chord tokens.
2. A Vendi score from chord-sequence embeddings produced by a contrastive reward model.

Generated intermediate files for each source tensor:
  - chord_frequency.json
      per-chord counts and probabilities.
  - diversity_metrics.json
      JSON summary including entropy statistics, Vendi score,
      embedding counts, and reward checkpoint metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from realchords.utils.chord_diversity_analysis import (
    DEFAULT_CHORD_NAMES,
    DEFAULT_CONFIG,
    analyze_diversity_file,
    load_contrastive_reward,
    resolve_device,
)
from realchords.utils.experiment_utils_reward_analysis import (
    build_hooktheory_tokenizer,
)
from realchords.utils.sequence_penalty_analysis import collect_sequence_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing generated .pt sequences (recursively searched)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to store diversity analysis outputs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="RL config file containing contrastive reward checkpoints",
    )
    parser.add_argument(
        "--contrastive_index",
        type=int,
        default=0,
        help="Which contrastive reward checkpoint to use (index into config list)",
    )
    parser.add_argument(
        "--chord_names_path",
        type=Path,
        default=DEFAULT_CHORD_NAMES,
        help="Path to chord_names JSON used by the tokenizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of sequences per embedding batch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device specifier or 'auto' to prefer CUDA when available",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of sequence tensors to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    tokenizer = build_hooktheory_tokenizer(args.chord_names_path)
    reward_wrapper, reward_model, checkpoint, device = load_contrastive_reward(
        args.config,
        args.contrastive_index,
        tokenizer,
        device,
    )

    sequence_files = collect_sequence_files(args.input_dir)
    if args.limit is not None:
        sequence_files = sequence_files[: args.limit]

    print(f"Loaded contrastive reward model from {checkpoint}")
    print(f"Analyzing {len(sequence_files)} tensors on device {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in sequence_files:
        analyze_diversity_file(
            path=path,
            output_dir=args.output_dir,
            input_root=args.input_dir,
            checkpoint=checkpoint,
            checkpoint_index=args.contrastive_index,
            tokenizer=tokenizer,
            reward_wrapper=reward_wrapper,
            reward_model=reward_model,
            device=device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
