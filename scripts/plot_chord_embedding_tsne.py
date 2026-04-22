#!/usr/bin/env python3
"""Plot a 2-D t-SNE of chord-sequence embeddings.

This script supports two input modes:

1. Direct groups: repeat `--group LABEL=PATH`, where PATH is either a generated
   `.pt` sequence tensor or a directory that contains generated `.pt` tensors.
2. Summary-derived groups: provide `--summary SUMMARY_JSON` together with one or
   more `--group_from_summary LABEL` arguments to load the source tensors listed
   for those systems in a summary JSON produced by
   `scripts/evaluate_generated_sequences.py`.

Example with direct folders:
    python scripts/plot_chord_embedding_tsne.py \
        --group "GAPT w/o Adv.=logs/generated/gapt_no_gail" \
        --group "GAPT=logs/generated/gapt" \
        --output_plot logs/figure4/ours_vs_ours_no_gail_chord_embedding_tsne.png

Example with a summary file:
    python scripts/plot_chord_embedding_tsne.py \
        --summary logs/test_eval/summary.json \
        --group_from_summary "GAPT w/o Adv." \
        --group_from_summary "GAPT" \
        --output_plot logs/figure4/ours_vs_ours_no_gail_chord_embedding_tsne.png

Outputs:
  - `--output_plot`: PNG scatter plot.
  - `--output_coordinates`: JSON with this format:
      {
        "labels": ["GAPT w/o Adv.", "GAPT", ...],
        "coordinates": [[x, y], ...]
      }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from realchords.utils.chord_diversity_analysis import (
    DEFAULT_CHORD_NAMES,
    DEFAULT_CONFIG,
    load_contrastive_reward,
    resolve_device,
)
from realchords.utils.experiment_utils_reward_analysis import (
    build_hooktheory_tokenizer,
)
from realchords.utils.sequence_penalty_analysis import (
    collect_sequence_files,
    load_sequences,
)

DEFAULT_PLOT_PATH = Path("logs/chord_embedding_tsne.png")
DEFAULT_COORDINATES_PATH = Path("logs/chord_embedding_tsne.json")
COLOR_OVERRIDES = {
    "GAPT": "#1f77b4",
    "GAPT w/o Adv.": "#d62728",
}
FALLBACK_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
FIGURE_SIZE = (8.0, 6.0)
POINT_SIZE = 12
POINT_ALPHA = 0.6
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 16
OUTPUT_DPI = 300


@dataclass(frozen=True)
class GroupSpec:
    label: str
    paths: List[Path]


@dataclass
class EmbeddingSummary:
    path: Path
    total_sequences: int
    embedded_sequences: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="Direct group specification in the form LABEL=PATH.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON produced by scripts/evaluate_generated_sequences.py.",
    )
    parser.add_argument(
        "--group_from_summary",
        action="append",
        default=None,
        help="System label to load from the summary file. Repeat as needed.",
    )
    parser.add_argument(
        "--output_plot",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path to the PNG plot.",
    )
    parser.add_argument(
        "--output_coordinates",
        type=Path,
        default=DEFAULT_COORDINATES_PATH,
        help="Path to the JSON file with t-SNE coordinates.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="RL config file listing contrastive reward checkpoints.",
    )
    parser.add_argument(
        "--contrastive_index",
        type=int,
        default=0,
        help="Which contrastive reward checkpoint to use.",
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
        help="Sequences per embedding batch.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity. It is reduced automatically if needed.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=13,
        help="Random seed for t-SNE.",
    )
    parser.add_argument(
        "--max_sequences_per_group",
        type=int,
        default=None,
        help="Optional cap on the number of sequences embedded per group.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    args = parser.parse_args()

    if not args.group and not args.group_from_summary:
        raise ValueError("Specify at least one --group or --group_from_summary")
    if args.group_from_summary and args.summary is None:
        raise ValueError(
            "--summary is required when using --group_from_summary"
        )
    return args


def parse_group(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            f"Invalid --group value '{value}'. Expected the form LABEL=PATH"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError("Group label cannot be empty")
    if not path.exists():
        raise FileNotFoundError(f"Group path does not exist: {path}")
    return label, path


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_paths_from_target(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix != ".pt":
            raise ValueError(f"Expected a .pt tensor file, got {path}")
        return [path]
    if path.is_dir():
        return collect_sequence_files(path)
    raise ValueError(f"Unsupported group target: {path}")


def collect_paths_from_summary(summary: dict, label: str) -> List[Path]:
    systems = summary.get("systems", {})
    system = systems.get(label)
    if system is None:
        raise KeyError(f"System '{label}' not found in summary")
    paths = []
    seen = set()
    for source in system.get("sources", []):
        path = Path(source["path"]).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Source path from summary does not exist: {path}"
            )
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No source tensors listed for '{label}'")
    return paths


def build_groups(args: argparse.Namespace) -> List[GroupSpec]:
    groups: List[GroupSpec] = []
    if args.group:
        for raw_group in args.group:
            label, path = parse_group(raw_group)
            groups.append(
                GroupSpec(label=label, paths=collect_paths_from_target(path))
            )
    if args.group_from_summary:
        summary = load_summary(args.summary.resolve())
        for label in args.group_from_summary:
            groups.append(
                GroupSpec(
                    label=label,
                    paths=collect_paths_from_summary(summary, label),
                )
            )
    if len(groups) < 2:
        raise ValueError(
            "At least two groups are required for a comparison plot"
        )
    return groups


def ensure_bos(batch: torch.Tensor, bos_token: int) -> torch.Tensor:
    if batch.size(1) == 0:
        return batch
    if torch.all(batch[:, 0] == bos_token):
        return batch
    if torch.all(batch[:, 0] != bos_token):
        bos_column = torch.full(
            (batch.size(0), 1), bos_token, dtype=batch.dtype
        )
        return torch.cat([bos_column, batch], dim=1)
    raise ValueError("Inconsistent BOS handling inside one batch")


def embed_sequences(
    paths: Sequence[Path],
    tokenizer,
    reward_wrapper,
    reward_model,
    device: torch.device,
    batch_size: int,
    max_sequences: int | None,
    verbose: bool = False,
) -> tuple[np.ndarray, List[EmbeddingSummary]]:
    embeddings: List[np.ndarray] = []
    summaries: List[EmbeddingSummary] = []
    pad = tokenizer.pad_token
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    remaining = max_sequences

    with torch.no_grad():
        for path in paths:
            sequences = load_sequences(path)
            total = int(sequences.size(0))
            embedded = 0
            if remaining is not None and remaining <= 0:
                summaries.append(
                    EmbeddingSummary(
                        path=path,
                        total_sequences=total,
                        embedded_sequences=0,
                    )
                )
                continue
            if remaining is not None and total > remaining:
                sequences = sequences[:remaining]
            effective_total = int(sequences.size(0))
            if verbose:
                print(f"Embedding {effective_total} sequences from {path}")

            for start in range(0, effective_total, batch_size):
                batch = sequences[start : start + batch_size]
                if batch.size(1) == 0:
                    continue
                batch = ensure_bos(batch, bos)
                model_tokens, _, model_mask, _ = (
                    reward_wrapper.get_inputs_from_sequence(batch)
                )
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
                embedded += int(non_empty.sum().item())

            summaries.append(
                EmbeddingSummary(
                    path=path,
                    total_sequences=effective_total,
                    embedded_sequences=embedded,
                )
            )
            if remaining is not None:
                remaining -= effective_total

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), summaries
    matrix = np.concatenate(embeddings, axis=0)
    return matrix.astype(np.float32, copy=False), summaries


def adjust_perplexity(requested: float, sample_count: int) -> float:
    if sample_count < 2:
        raise ValueError("Need at least two embeddings for t-SNE")
    max_valid = max(1.0, float(sample_count - 1))
    if requested >= max_valid:
        return max(1.0, min(requested, max_valid - 1e-6))
    return requested


def run_tsne(
    embeddings: np.ndarray, perplexity: float, random_state: int
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="random",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def save_coordinates(
    coordinates: np.ndarray, labels: Sequence[str], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": list(labels),
        "coordinates": coordinates.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def plot_groups(
    coordinates: np.ndarray,
    labels: Sequence[str],
    ordered_labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_array = np.asarray(labels)
    plt.figure(figsize=FIGURE_SIZE)
    for idx, label in enumerate(ordered_labels):
        mask = label_array == label
        if not np.any(mask):
            continue
        coords = coordinates[mask]
        color = COLOR_OVERRIDES.get(
            label, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
        )
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            label=label,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
            c=color,
        )
    if title:
        plt.title(title, fontsize=LEGEND_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=OUTPUT_DPI)
    plt.close()


def summarize(label: str, summaries: Sequence[EmbeddingSummary]) -> None:
    total_sequences = sum(item.total_sequences for item in summaries)
    embedded_sequences = sum(item.embedded_sequences for item in summaries)
    print(
        f"{label}: {embedded_sequences}/{total_sequences} sequences produced embeddings across {len(summaries)} files"
    )


def main() -> None:
    torch.set_grad_enabled(False)
    args = parse_args()
    groups = build_groups(args)

    tokenizer = build_hooktheory_tokenizer(args.chord_names_path.resolve())
    requested_device = resolve_device(args.device)
    reward_wrapper, reward_model, checkpoint, device = load_contrastive_reward(
        args.config.resolve(),
        args.contrastive_index,
        tokenizer,
        requested_device,
    )
    reward_model.eval()

    embeddings_by_label: Dict[str, np.ndarray] = {}
    ordered_labels: List[str] = []
    for group in groups:
        matrix, summaries = embed_sequences(
            group.paths,
            tokenizer,
            reward_wrapper,
            reward_model,
            device,
            args.batch_size,
            args.max_sequences_per_group,
            verbose=args.verbose,
        )
        if matrix.size == 0:
            raise ValueError(f"No embeddings extracted for {group.label}")
        ordered_labels.append(group.label)
        embeddings_by_label[group.label] = matrix
        summarize(group.label, summaries)

    stacked_embeddings = np.concatenate(
        [embeddings_by_label[label] for label in ordered_labels],
        axis=0,
    )
    point_labels = [
        label
        for label in ordered_labels
        for _ in range(embeddings_by_label[label].shape[0])
    ]

    adjusted_perplexity = adjust_perplexity(
        args.perplexity, stacked_embeddings.shape[0]
    )
    if adjusted_perplexity != args.perplexity:
        print(
            f"Adjusted perplexity from {args.perplexity} to {adjusted_perplexity:.3f} for {stacked_embeddings.shape[0]} samples"
        )

    coordinates = run_tsne(
        stacked_embeddings, adjusted_perplexity, args.random_state
    )
    plot_groups(
        coordinates, point_labels, ordered_labels, args.output_plot, args.title
    )
    save_coordinates(coordinates, point_labels, args.output_coordinates)

    print(f"Saved t-SNE plot to {args.output_plot.resolve()}")
    print(f"Saved t-SNE coordinates to {args.output_coordinates.resolve()}")
    print(f"Contrastive checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
