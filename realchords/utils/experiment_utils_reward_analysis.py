"""Utilities for reward-analysis scripts.

This module keeps only the helpers required by the public evaluation scripts to:

- build the Hooktheory tokenizer used by reward checkpoints,
- read reward checkpoint paths from an RL config, and
- resolve checkpoint paths relative to the repository root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer

DEFAULT_CHORD_NAMES_PATH = Path("data/cache/chord_names_augmented.json")


def _load_yaml_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_hooktheory_tokenizer(
    chord_names_path: Path | None = None,
) -> HooktheoryTokenizer:
    """Instantiate the Hooktheory tokenizer used across reward models."""

    chord_names_path = chord_names_path or DEFAULT_CHORD_NAMES_PATH
    with chord_names_path.open("r", encoding="utf-8") as handle:
        chord_names = json.load(handle)
    return HooktheoryTokenizer(chord_names=chord_names)


def extract_reward_paths(config_path: Path) -> Dict[str, List[str]]:
    """Return reward checkpoint paths grouped by reward family."""

    config = _load_yaml_config(config_path)
    keys = {
        "contrastive": "contrastive_reward_model_path",
        "discriminative": "discriminative_reward_model_path",
        "contrastive_rhythm": "contrastive_reward_rhythm_model_path",
        "discriminative_rhythm": "discriminative_reward_rhythm_model_path",
    }
    reward_paths: Dict[str, List[str]] = {}
    for group, key in keys.items():
        values = config.get(key, [])
        if isinstance(values, str):
            reward_paths[group] = [values]
        elif isinstance(values, Iterable):
            reward_paths[group] = [str(item) for item in values]
        else:
            reward_paths[group] = []
    return reward_paths


def extract_model_part(config_path: Path) -> str:
    """Read the reward target part from an RL config, defaulting to chord."""

    config = _load_yaml_config(config_path)
    model_part = config.get("model_part")
    if not isinstance(model_part, str):
        model_part = "chord"
    return model_part


def _resolve_checkpoint_path(model_path: str, config_path: Path) -> Path:
    """Resolve checkpoint path relative to the repository root."""

    path = Path(model_path).expanduser()
    if path.is_absolute() and path.exists():
        return path

    repo_root = config_path.resolve().parents[2]
    candidate = (repo_root / path).resolve()
    if candidate.exists():
        return candidate

    return path
