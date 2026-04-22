"""Shared helpers for rule-based penalty and harmony analysis on generated sequences.

This module is the reusable core behind `scripts/analyze_sequence_penalties.py` and
the combined `scripts/evaluate_generated_sequences.py` entry point. Keeping these
helpers inside the package (rather than in a sibling script) means other scripts
and downstream users can import them without relying on the `scripts/` directory
being on `sys.path`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Sequence

import torch

from realchords.dataset.hooktheory_tokenizer import (
    HooktheoryTokenizer,
    to_midi_pitch,
)
from realchords.utils.eval_utils import evaluate_note_in_chord_ratio


MODEL_PART_CHOICES = ("chord", "melody")
SEQUENCE_ORDER_CHOICES = ("chord_first", "melody_first")


@dataclass(frozen=True)
class AnalysisConfig:
    input_dir: Path
    output_dir: Path
    chord_names_path: Path
    model_part: str
    long_note_threshold: int
    repetition_threshold: int
    max_frames: int
    frames_per_beat: int = 4
    sequence_order: str = "chord_first"

    @property
    def tokens_per_frame(self) -> int:
        return 2

    @property
    def tokens_per_beat(self) -> int:
        return self.frames_per_beat * self.tokens_per_frame

    @property
    def max_beats(self) -> int:
        return self.max_frames // self.frames_per_beat


def load_tokenizer(chord_names_path: Path) -> HooktheoryTokenizer:
    with chord_names_path.open("r", encoding="utf-8") as handle:
        chord_names = json.load(handle)
    return HooktheoryTokenizer(chord_names=chord_names)


def collect_sequence_files(input_dir: Path) -> list[Path]:
    """Return all candidate generated sequence tensors under input_dir."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    candidates: list[Path] = []
    for path in sorted(input_dir.rglob("*.pt")):
        name = path.name
        if "kl" in name:
            continue
        if "melody_order" in name:
            continue
        if "generated" not in name:
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No generated sequence tensors found in {input_dir}")
    return candidates


def ensure_output_folder(output_dir: Path, input_file: Path, root: Path) -> Path:
    relative = input_file.relative_to(root)
    folder = output_dir / relative.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def strip_bos(sequences: torch.Tensor, tokenizer: HooktheoryTokenizer) -> torch.Tensor:
    if sequences.size(1) < 1:
        raise ValueError("Sequences must include at least the BOS token")
    bos_id = tokenizer.bos_token
    has_bos = (sequences[:, 0] == bos_id).all()
    if not has_bos:
        return sequences
    return sequences[:, 1:]


def split_tokens_by_order(
    sequence: torch.Tensor, sequence_order: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_order == "chord_first":
        return sequence[::2], sequence[1::2]
    if sequence_order == "melody_first":
        return sequence[1::2], sequence[::2]
    raise ValueError(f"Unsupported sequence_order: {sequence_order}")


def get_model_tokens(
    sequence: torch.Tensor, model_part: str, sequence_order: str
) -> torch.Tensor:
    chord_tokens, melody_tokens = split_tokens_by_order(sequence, sequence_order)
    if model_part == "chord":
        return chord_tokens
    if model_part == "melody":
        return melody_tokens
    raise ValueError(f"Unsupported model_part: {model_part}")


def compute_long_note_penalty(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    threshold: int,
    sequence_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    penalties = []
    counts = []
    pad_token = tokenizer.pad_token
    for seq in sequences:
        seq = seq.cpu()
        model_tokens = get_model_tokens(seq, model_part, sequence_order).tolist()
        penalty = 0
        count = 0
        for token, group in groupby(model_tokens):
            if token == pad_token:
                break
            group_len = len(list(group))
            if group_len > threshold - 1:
                penalty -= group_len
                count += group_len
        penalties.append(penalty)
        counts.append(count)
    return (
        torch.tensor(penalties, dtype=torch.float32),
        torch.tensor(counts, dtype=torch.int64),
    )


def decode_repetition_units(
    model_tokens: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
) -> Sequence[int] | Sequence[str]:
    if model_part == "chord":
        annotations = tokenizer.decode_chord_frames(model_tokens)
        return [ann["chord_name"] for ann in annotations]
    if model_part == "melody":
        annotations = tokenizer.decode_melody_frames(model_tokens)
        return [
            to_midi_pitch(ann["octave"], ann["pitch_class"]) for ann in annotations
        ]
    raise ValueError(f"Unsupported model_part: {model_part}")


def compute_repetition_penalty(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    threshold: int,
    sequence_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    penalties = []
    counts = []
    special_tokens = {
        tokenizer.pad_token,
        tokenizer.bos_token,
        tokenizer.eos_token,
    }
    if hasattr(tokenizer, "silence_token"):
        special_tokens.add(tokenizer.silence_token)

    for seq in sequences:
        seq = seq.cpu()
        model_tokens = get_model_tokens(seq, model_part, sequence_order)
        penalty = 0
        count = 0
        try:
            gen_list = decode_repetition_units(model_tokens, tokenizer, model_part)
            for _, group in groupby(gen_list):
                group_len = len(list(group))
                if group_len > threshold:
                    penalty -= group_len
                    count += group_len
        except Exception:
            token_list = model_tokens.tolist()
            for token, group in groupby(token_list):
                if token in special_tokens:
                    continue
                group_len = len(list(group))
                if group_len > threshold:
                    penalty -= group_len
                    count += group_len
        penalties.append(penalty)
        counts.append(count)
    return (
        torch.tensor(penalties, dtype=torch.float32),
        torch.tensor(counts, dtype=torch.int64),
    )


def compute_per_beat_ratios(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    config: AnalysisConfig,
) -> torch.Tensor:
    batch_size, _ = sequences.shape
    per_beat = torch.full(
        (batch_size, config.max_beats),
        float("nan"),
        dtype=torch.float32,
    )
    beat_len = config.tokens_per_beat

    for beat_idx in range(config.max_beats):
        start = beat_idx * beat_len
        end = start + beat_len
        window_batch = torch.full(
            (batch_size, beat_len),
            tokenizer.pad_token,
            dtype=sequences.dtype,
        )
        for row_idx in range(batch_size):
            seq = sequences[row_idx]
            if start >= seq.size(0):
                continue
            slice_tokens = seq[start:end]
            window_batch[row_idx, : slice_tokens.size(0)] = slice_tokens
        ratios = evaluate_note_in_chord_ratio(
            window_batch,
            tokenizer,
            model_part=model_part,
            sequence_order=config.sequence_order,
        )
        per_beat[:, beat_idx] = ratios
    return per_beat


def load_sequences(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        if data.dim() != 2:
            raise ValueError(
                f"Expected rank-2 tensor in {path}, got shape {tuple(data.shape)}"
            )
        return data.long()
    if isinstance(data, dict):
        for key in ("sequences", "generated", "decoder_preds"):
            if key in data and isinstance(data[key], torch.Tensor):
                tensor = data[key].long()
                if tensor.dim() != 2:
                    raise ValueError(
                        f"Expected rank-2 tensor for key '{key}' in {path}, got shape {tuple(tensor.shape)}"
                    )
                return tensor
    raise TypeError(f"Unsupported sequence container in {path}: {type(data)!r}")


def analyze_penalty_file(
    path: Path,
    config: AnalysisConfig,
    tokenizer: HooktheoryTokenizer,
) -> None:
    """Compute note-in-chord ratios and rule-based penalties for one tensor file."""
    print(f"Processing {path}")
    sequences = load_sequences(path)
    sequences = strip_bos(sequences, tokenizer)

    overall_ratio = evaluate_note_in_chord_ratio(
        sequences,
        tokenizer,
        model_part=config.model_part,
        sequence_order=config.sequence_order,
    ).float()

    per_beat_ratio = compute_per_beat_ratios(
        sequences,
        tokenizer,
        config.model_part,
        config,
    )

    long_penalty, long_count = compute_long_note_penalty(
        sequences,
        tokenizer,
        config.model_part,
        config.long_note_threshold,
        config.sequence_order,
    )

    repetition_penalty, repetition_count = compute_repetition_penalty(
        sequences,
        tokenizer,
        config.model_part,
        config.repetition_threshold,
        config.sequence_order,
    )

    penalties_out = {
        "long_note_penalty": long_penalty,
        "long_note_count": long_count,
        "repetition_penalty": repetition_penalty,
        "repetition_count": repetition_count,
    }

    output_folder = ensure_output_folder(config.output_dir, path, config.input_dir)
    torch.save(overall_ratio, output_folder / "note_in_chord_ratio.pt")
    torch.save(
        {"per_beat_ratio": per_beat_ratio},
        output_folder / "per_beat_note_in_chord_ratio.pt",
    )
    torch.save(penalties_out, output_folder / "penalties.pt")
