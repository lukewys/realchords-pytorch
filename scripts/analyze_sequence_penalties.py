#!/usr/bin/env python3
"""Analyze note-in-chord ratios and rule-based penalties for generated sequences.

This script recursively scans an input directory for generated sequence tensors and
writes per-file intermediate artifacts under a mirrored output directory.

Generated intermediate files for each source tensor:
  - note_in_chord_ratio.pt
      1-D float tensor with one harmony score per sequence.
  - per_beat_note_in_chord_ratio.pt
      dict containing `per_beat_ratio`, a [num_sequences, max_beats] float tensor.
  - penalties.pt
      dict with `long_note_penalty`, `long_note_count`,
      `repetition_penalty`, and `repetition_count` tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from realchords.utils.sequence_penalty_analysis import (
    AnalysisConfig,
    MODEL_PART_CHOICES,
    SEQUENCE_ORDER_CHOICES,
    analyze_penalty_file,
    collect_sequence_files,
    load_tokenizer,
)


def parse_args() -> AnalysisConfig:
    repo_root = Path(__file__).resolve().parents[1]
    default_chord_names = (
        repo_root / "data" / "cache" / "chord_names_augmented.json"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing generated .pt sequences",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory to store analysis outputs"
    )
    parser.add_argument(
        "--chord_names_path",
        type=Path,
        default=default_chord_names,
        help="Path to chord_names JSON (default: data/cache/chord_names_augmented.json)",
    )
    parser.add_argument(
        "--model_part",
        type=str,
        default="chord",
        choices=MODEL_PART_CHOICES,
        help="Model part used for generation (determines token slicing)",
    )
    parser.add_argument(
        "--long_note_threshold",
        type=int,
        default=32,
        help="Frame threshold (inclusive) for long-note penalties",
    )
    parser.add_argument(
        "--repetition_threshold",
        type=int,
        default=4,
        help="Consecutive identical event threshold for repetition penalty",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=256,
        help="Maximum number of frames per sequence (used for padding beats)",
    )
    parser.add_argument(
        "--frames_per_beat",
        type=int,
        default=4,
        help="Number of frames per beat",
    )
    parser.add_argument(
        "--sequence_order",
        type=str,
        default="chord_first",
        choices=SEQUENCE_ORDER_CHOICES,
        help="Ordering of tokens within a frame.",
    )

    args = parser.parse_args()

    if args.frames_per_beat <= 0:
        raise ValueError("frames_per_beat must be positive")
    if args.max_frames <= 0:
        raise ValueError("max_frames must be positive")

    return AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chord_names_path=args.chord_names_path,
        model_part=args.model_part,
        long_note_threshold=args.long_note_threshold,
        repetition_threshold=args.repetition_threshold,
        max_frames=args.max_frames,
        frames_per_beat=args.frames_per_beat,
        sequence_order=args.sequence_order,
    )


def main() -> None:
    config = parse_args()
    tokenizer = load_tokenizer(config.chord_names_path)

    sequence_files = collect_sequence_files(config.input_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for path in sequence_files:
        analyze_penalty_file(path, config, tokenizer)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
