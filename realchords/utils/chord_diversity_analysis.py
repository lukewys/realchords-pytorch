"""Shared helpers for chord diversity analysis on generated sequences.

Used by `scripts/analyze_chord_diversity.py`, `scripts/evaluate_generated_sequences.py`,
and `scripts/plot_chord_embedding_tsne.py`. Keeping this logic inside the package lets
other scripts import it without relying on `scripts/` being on `sys.path`.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch

from realchords.lit_module.contrastive_reward import LitContrastiveReward
from realchords.rl.reward.model_based_rewards import ContrastiveRewardFn
from realchords.utils.experiment_utils_reward_analysis import (
    _resolve_checkpoint_path,
    extract_model_part,
    extract_reward_paths,
)
from realchords.utils.inference_utils import load_lit_model
from realchords.utils.sequence_penalty_analysis import (
    ensure_output_folder,
    load_sequences,
)
from vendi_score.vendi import score_X


DEFAULT_CONFIG = Path("configs/single_agent_rl/realchords.yml")
DEFAULT_CHORD_NAMES = Path("data/cache/chord_names_augmented.json")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_contrastive_reward(
    config_path: Path,
    index: int,
    tokenizer,
    device: torch.device,
) -> tuple[ContrastiveRewardFn, torch.nn.Module, str, torch.device]:
    reward_paths = extract_reward_paths(config_path)
    contrastive_paths = reward_paths.get("contrastive", [])
    if not contrastive_paths:
        raise ValueError(
            f"No contrastive reward checkpoints listed in {config_path}"
        )
    if index < 0 or index >= len(contrastive_paths):
        raise IndexError(
            f"contrastive-index {index} out of range (found {len(contrastive_paths)} checkpoints)"
        )

    model_part = extract_model_part(config_path)
    if model_part != "chord":
        raise ValueError(
            "Contrastive reward config must target the chord part for this analysis"
        )

    checkpoint = _resolve_checkpoint_path(contrastive_paths[index], config_path)
    model = load_lit_model(
        model_path=str(checkpoint),
        lit_module_cls=LitContrastiveReward,
        compile=False,
        return_only_model=True,
    )
    actual_device = device
    try:
        model.to(actual_device)
    except RuntimeError as exc:
        message = str(exc).lower()
        if actual_device.type == "cuda" and "out of memory" in message:
            print("CUDA OOM when loading reward model; falling back to CPU")
            actual_device = torch.device("cpu")
            model.to(actual_device)
        else:
            raise
    model.eval()
    model.device = actual_device

    wrapper = ContrastiveRewardFn(
        model=model,
        pad_token_id=tokenizer.pad_token,
        bos_token_id=tokenizer.bos_token,
        eos_token_id=tokenizer.eos_token,
        model_part=model_part,
    )
    return wrapper, model, str(checkpoint), actual_device


def chord_name_from_token(token_id: int, tokenizer) -> str | None:
    name = tokenizer.id_to_name.get(int(token_id))
    if not name:
        return None
    if name.startswith("CHORD_ON_"):
        return name.split("CHORD_ON_")[1]
    if name.startswith("CHORD_"):
        return name.split("CHORD_")[1]
    return None


def accumulate_chord_counts(
    model_tokens: torch.Tensor,
    tokenizer,
    counter: Counter,
) -> None:
    pad = tokenizer.pad_token
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    for row in model_tokens:
        for token in row.tolist():
            if token == pad:
                break
            if token == eos:
                break
            if token == bos:
                continue
            chord_name = chord_name_from_token(token, tokenizer)
            if chord_name is not None:
                counter[chord_name] += 1


def compute_entropy(counter: Counter, tokenizer) -> tuple[float, float, float]:
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0, 0.0

    probs = [count / total for count in counter.values() if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)

    num_observed = len(probs)
    observed_norm = entropy / math.log(num_observed) if num_observed > 1 else 0.0

    max_possible = math.log(max(len(tokenizer.chord_names), 1))
    normalized = entropy / max_possible if max_possible > 0 else 0.0

    return entropy, normalized, observed_norm


def compute_vendi_score(embeddings: List[np.ndarray]) -> float | None:
    if not embeddings:
        return None
    matrix = np.concatenate(embeddings, axis=0)
    if matrix.shape[0] < 2:
        return 1.0
    return float(score_X(matrix, normalize=True))


def analyze_diversity_file(
    path: Path,
    output_dir: Path,
    input_root: Path,
    checkpoint: str,
    checkpoint_index: int,
    tokenizer,
    reward_wrapper: ContrastiveRewardFn,
    reward_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> None:
    """Write per-file chord frequency and diversity metrics for one tensor file."""
    print(f"Processing {path}")
    sequences = load_sequences(path)

    chord_counter: Counter = Counter()
    embeddings: List[np.ndarray] = []

    total_sequences = sequences.size(0)
    valid_embedding_sequences = 0
    pad = tokenizer.pad_token
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token

    with torch.no_grad():
        for start in range(0, total_sequences, batch_size):
            batch = sequences[start : start + batch_size]
            if batch.size(1) == 0:
                continue
            if not torch.all(batch[:, 0] == bos):
                raise ValueError(
                    f"Sequence tensor {path} is missing BOS tokens in some rows"
                )
            model_tokens, _, model_mask, _ = reward_wrapper.get_inputs_from_sequence(
                batch
            )
            accumulate_chord_counts(model_tokens, tokenizer, chord_counter)

            chord_mask = (
                (model_tokens != pad)
                & (model_tokens != bos)
                & (model_tokens != eos)
            )
            lengths = chord_mask.sum(dim=1)
            non_empty = lengths > 0
            if not non_empty.any():
                continue

            tokens_to_embed = model_tokens[non_empty].to(device)
            mask_to_embed = model_mask[non_empty].to(device)

            embed_batch = reward_model.get_chord_embed(
                chord=tokens_to_embed,
                chord_mask=mask_to_embed,
            )
            embeddings.append(embed_batch.cpu().numpy())
            valid_embedding_sequences += int(non_empty.sum().item())

    total_chord_frames = int(sum(chord_counter.values()))

    entropy, normalized_entropy, observed_norm = compute_entropy(
        chord_counter, tokenizer
    )
    vendi_score = compute_vendi_score(embeddings)
    embedding_dim = int(embeddings[0].shape[1]) if embeddings else None

    output_folder = ensure_output_folder(output_dir, path, root=input_root)

    freq_path = output_folder / "chord_frequency.json"
    total_count = sum(chord_counter.values())
    frequencies = {
        chord: {
            "count": int(count),
            "probability": (count / total_count) if total_count else 0.0,
        }
        for chord, count in sorted(
            chord_counter.items(), key=lambda item: (-item[1], item[0])
        )
    }
    with freq_path.open("w", encoding="utf-8") as handle:
        json.dump(frequencies, handle, indent=2, sort_keys=True)

    summary_path = output_folder / "diversity_metrics.json"
    summary = {
        "input_file": str(path),
        "num_sequences": int(total_sequences),
        "num_sequences_with_embeddings": int(valid_embedding_sequences),
        "total_chord_frames": int(total_chord_frames),
        "num_unique_chords": len(chord_counter),
        "entropy_nats": entropy,
        "normalized_entropy_all_chords": normalized_entropy,
        "normalized_entropy_observed": observed_norm,
        "vendi_score": vendi_score,
        "contrastive_checkpoint": checkpoint,
        "contrastive_index": checkpoint_index,
        "device": str(device),
        "embedding_dim": embedding_dim,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
