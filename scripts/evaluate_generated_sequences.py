#!/usr/bin/env python3
"""Unified harmony and diversity evaluation for generated sequence folders.

This script is the recommended public entry point for reproducing the Figure 4
evaluation workflow from user-generated checkpoints.

Usage example:
    python scripts/evaluate_generated_sequences.py \
        --system "Online MLE=logs/generated/online_mle" \
        --system "ReaLchords=logs/generated/realchords" \
        --system "GAPT w/o Adv.=logs/generated/gapt_no_gail" \
        --system "GAPT=logs/generated/gapt" \
        --analysis_root logs/figure4_eval \
        --summary_path logs/figure4_eval/summary.json \
        --config configs/single_agent_rl/realchords.yml

Inputs:
  Each `--system` value must be `LABEL=DIR`, where DIR contains `.pt` files
  produced by `scripts/generate_sequences.py`.

Outputs:
  - Per-file intermediate artifacts under:
      <analysis_root>/<system-slug>/penalties/...
      <analysis_root>/<system-slug>/diversity/...
  - A system-level summary JSON at `--summary_path` with this shape:
      {
        "systems": {
          "Label": {
            "input_dir": "...",
            "num_sequence_files": 3,
            "num_sequences": 1024,
            "overall_note_in_chord_ratio": 0.48,
            "overall_vendi_score": 21.7,
            "sources": [{"path": "...", "num_sequences": 256}, ...]
          }
        }
      }
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from realchords.utils.chord_diversity_analysis import (
    DEFAULT_CHORD_NAMES,
    DEFAULT_CONFIG,
    accumulate_chord_counts,
    analyze_diversity_file,
    compute_entropy,
    compute_vendi_score,
    load_contrastive_reward,
    resolve_device,
)
from realchords.utils.eval_utils import evaluate_note_in_chord_ratio
from realchords.utils.experiment_utils_reward_analysis import (
    build_hooktheory_tokenizer,
)
from realchords.utils.sequence_penalty_analysis import (
    AnalysisConfig,
    MODEL_PART_CHOICES,
    SEQUENCE_ORDER_CHOICES,
    analyze_penalty_file,
    collect_sequence_files,
    load_sequences,
    load_tokenizer,
    strip_bos,
)


@dataclass(frozen=True)
class SystemSpec:
    label: str
    directory: Path
    slug: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "system"


def parse_system_arg(raw: str) -> SystemSpec:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --system value '{raw}'. Expected the form LABEL=DIR"
        )
    label, directory = raw.split("=", 1)
    label = label.strip()
    path = Path(directory).expanduser().resolve()
    if not label:
        raise ValueError("System label cannot be empty")
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"System directory not found: {path}")
    return SystemSpec(label=label, directory=path, slug=slugify(label))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="System specification in the form LABEL=DIR. Repeat for multiple systems.",
    )
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=Path("logs/generated_sequence_eval"),
        help="Directory where per-file intermediate analysis artifacts will be written.",
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("logs/generated_sequence_eval/summary.json"),
        help="Path of the combined system-level summary JSON.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="RL config file listing contrastive reward checkpoints for Vendi embeddings.",
    )
    parser.add_argument(
        "--contrastive_index",
        type=int,
        default=0,
        help="Which contrastive reward checkpoint to use for Vendi embeddings.",
    )
    parser.add_argument(
        "--chord_names_path",
        type=Path,
        default=DEFAULT_CHORD_NAMES,
        help="Tokenizer chord-name mapping JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device specifier or 'auto'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Embedding batch size for Vendi score computation.",
    )
    parser.add_argument(
        "--model_part",
        type=str,
        default="chord",
        choices=MODEL_PART_CHOICES,
        help="Model part used for the harmony analysis.",
    )
    parser.add_argument(
        "--sequence_order",
        type=str,
        default="chord_first",
        choices=SEQUENCE_ORDER_CHOICES,
        help="Ordering of tokens within a frame for harmony analysis.",
    )
    parser.add_argument(
        "--long_note_threshold",
        type=int,
        default=32,
        help="Frame threshold (inclusive) for long-note penalties.",
    )
    parser.add_argument(
        "--repetition_threshold",
        type=int,
        default=4,
        help="Consecutive identical event threshold for repetition penalty.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=256,
        help="Maximum number of frames per sequence for per-beat padding.",
    )
    parser.add_argument(
        "--frames_per_beat",
        type=int,
        default=4,
        help="Frames per beat used during harmony analysis.",
    )
    parser.add_argument(
        "--skip_intermediate_artifacts",
        action="store_true",
        help="If set, compute only the system summary and skip writing per-file artifacts.",
    )
    return parser.parse_args()


def build_penalty_config(
    args: argparse.Namespace, input_dir: Path, output_dir: Path
) -> AnalysisConfig:
    return AnalysisConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        chord_names_path=args.chord_names_path,
        model_part=args.model_part,
        long_note_threshold=args.long_note_threshold,
        repetition_threshold=args.repetition_threshold,
        max_frames=args.max_frames,
        frames_per_beat=args.frames_per_beat,
        sequence_order=args.sequence_order,
    )


def accumulate_harmony_metrics(
    sequence_files: List[Path],
    tokenizer,
    model_part: str,
    sequence_order: str,
) -> Dict[str, object]:
    total_sequences = 0
    total_valid_frames = 0
    total_correct_frames = 0
    sources = []

    for path in sequence_files:
        sequences = load_sequences(path)
        total_sequences += int(sequences.size(0))
        sources.append(
            {"path": str(path), "num_sequences": int(sequences.size(0))}
        )
        sequences = strip_bos(sequences, tokenizer)
        _, valid_counts, correct_counts = evaluate_note_in_chord_ratio(
            sequences,
            tokenizer,
            model_part=model_part,
            return_count=True,
            sequence_order=sequence_order,
        )
        total_valid_frames += int(valid_counts.sum().item())
        total_correct_frames += int(correct_counts.sum().item())

    overall_ratio = (
        float(total_correct_frames / total_valid_frames)
        if total_valid_frames
        else None
    )
    return {
        "num_sequences": total_sequences,
        "total_valid_frames": total_valid_frames,
        "total_correct_frames": total_correct_frames,
        "overall_note_in_chord_ratio": overall_ratio,
        "sources": sources,
    }


def accumulate_diversity_metrics(
    sequence_files: List[Path],
    tokenizer,
    reward_wrapper,
    reward_model,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    chord_counter: Counter = Counter()
    embeddings: List[np.ndarray] = []
    total_sequences = 0
    total_embedded_sequences = 0
    pad = tokenizer.pad_token
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    with torch.no_grad():
        for path in sequence_files:
            sequences = load_sequences(path)
            total_sequences += int(sequences.size(0))
            for start in range(0, sequences.size(0), batch_size):
                batch = sequences[start : start + batch_size]
                if batch.size(1) == 0:
                    continue
                if torch.all(batch[:, 0] != bos):
                    bos_column = torch.full(
                        (batch.size(0), 1),
                        bos,
                        dtype=batch.dtype,
                    )
                    batch = torch.cat([bos_column, batch], dim=1)
                elif not torch.all(batch[:, 0] == bos):
                    raise ValueError(
                        f"Inconsistent BOS handling in {path}; either all rows should contain BOS or none should."
                    )

                model_tokens, _, model_mask, _ = (
                    reward_wrapper.get_inputs_from_sequence(batch)
                )
                accumulate_chord_counts(model_tokens, tokenizer, chord_counter)

                chord_mask = (
                    (model_tokens != pad)
                    & (model_tokens != bos)
                    & (model_tokens != eos)
                )
                non_empty = chord_mask.sum(dim=1) > 0
                if not non_empty.any():
                    continue

                tokens_to_embed = model_tokens[non_empty].to(device)
                mask_to_embed = model_mask[non_empty].to(device)
                embed_batch = reward_model.get_chord_embed(
                    chord=tokens_to_embed,
                    chord_mask=mask_to_embed,
                )
                embeddings.append(embed_batch.cpu().numpy())
                total_embedded_sequences += int(non_empty.sum().item())

    entropy, normalized_entropy, observed_norm = compute_entropy(
        chord_counter, tokenizer
    )
    vendi_score = compute_vendi_score(embeddings)
    return {
        "num_sequences": total_sequences,
        "num_sequences_with_embeddings": total_embedded_sequences,
        "total_chord_frames": int(sum(chord_counter.values())),
        "num_unique_chords": len(chord_counter),
        "entropy_nats": float(entropy),
        "normalized_entropy_all_chords": float(normalized_entropy),
        "normalized_entropy_observed": float(observed_norm),
        "overall_vendi_score": vendi_score,
    }


def main() -> None:
    args = parse_args()
    systems = [parse_system_arg(item) for item in args.system]
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = build_hooktheory_tokenizer(args.chord_names_path)
    harmony_tokenizer = load_tokenizer(args.chord_names_path)
    device = resolve_device(args.device)
    reward_wrapper, reward_model, checkpoint, device = load_contrastive_reward(
        args.config,
        args.contrastive_index,
        tokenizer,
        device,
    )

    summary: Dict[str, object] = {
        "analysis_root": str(args.analysis_root.resolve()),
        "config": str(Path(args.config).resolve()),
        "contrastive_index": int(args.contrastive_index),
        "contrastive_checkpoint": checkpoint,
        "model_part": args.model_part,
        "sequence_order": args.sequence_order,
        "systems": {},
    }

    for system in systems:
        print(f"Evaluating {system.label} from {system.directory}")
        sequence_files = collect_sequence_files(system.directory)
        penalties_dir = args.analysis_root / system.slug / "penalties"
        diversity_dir = args.analysis_root / system.slug / "diversity"

        if not args.skip_intermediate_artifacts:
            penalty_config = build_penalty_config(
                args, system.directory, penalties_dir
            )
            penalties_dir.mkdir(parents=True, exist_ok=True)
            diversity_dir.mkdir(parents=True, exist_ok=True)
            for path in sequence_files:
                analyze_penalty_file(path, penalty_config, harmony_tokenizer)
                analyze_diversity_file(
                    path=path,
                    output_dir=diversity_dir,
                    input_root=system.directory,
                    checkpoint=checkpoint,
                    checkpoint_index=args.contrastive_index,
                    tokenizer=tokenizer,
                    reward_wrapper=reward_wrapper,
                    reward_model=reward_model,
                    device=device,
                    batch_size=args.batch_size,
                )

        harmony_metrics = accumulate_harmony_metrics(
            sequence_files,
            tokenizer=harmony_tokenizer,
            model_part=args.model_part,
            sequence_order=args.sequence_order,
        )
        diversity_metrics = accumulate_diversity_metrics(
            sequence_files,
            tokenizer=tokenizer,
            reward_wrapper=reward_wrapper,
            reward_model=reward_model,
            device=device,
            batch_size=args.batch_size,
        )

        summary["systems"][system.label] = {
            "input_dir": str(system.directory),
            "num_sequence_files": len(sequence_files),
            "num_sequences": harmony_metrics["num_sequences"],
            "num_sequences_with_embeddings": diversity_metrics[
                "num_sequences_with_embeddings"
            ],
            "total_valid_frames": harmony_metrics["total_valid_frames"],
            "total_correct_frames": harmony_metrics["total_correct_frames"],
            "overall_note_in_chord_ratio": harmony_metrics[
                "overall_note_in_chord_ratio"
            ],
            "overall_vendi_score": diversity_metrics["overall_vendi_score"],
            "entropy_nats": diversity_metrics["entropy_nats"],
            "normalized_entropy_all_chords": diversity_metrics[
                "normalized_entropy_all_chords"
            ],
            "normalized_entropy_observed": diversity_metrics[
                "normalized_entropy_observed"
            ],
            "total_chord_frames": diversity_metrics["total_chord_frames"],
            "num_unique_chords": diversity_metrics["num_unique_chords"],
            "sources": harmony_metrics["sources"],
            "analysis_dirs": {
                "penalties": str(penalties_dir),
                "diversity": str(diversity_dir),
            },
        }

    with args.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote summary to {args.summary_path.resolve()}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
