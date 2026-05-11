from __future__ import annotations

import importlib.util
import json
import unittest
from functools import lru_cache, partial
from pathlib import Path

import torch

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.model.gen_model import (
    DecoderTransformer,
    EncoderDecoderTransformer,
)
from realchords.model.sampling import (
    filter_invalid_tokens_generate_single_part,
    filter_invalid_tokens_generate_unconditional,
)
from realchords.realjam.agent_interface import (
    MODEL_PATH_CANDIDATES,
    OFFLINE_MODEL_CANDIDATES,
    ONLINE_MODEL_CANDIDATES,
    _resolve_existing_path,
)
from realchords.utils.inference_utils import (
    load_gen_model_from_state_dict,
    load_rl_model,
)


MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

if MLX_AVAILABLE:
    from realchords.realjam.mlx_models.offline_model import (
        MLXEncoderDecoderTransformer,
    )
    from realchords.realjam.mlx_models.online_model import MLXDecoderTransformer


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


@lru_cache(maxsize=1)
def _tokenizer() -> HooktheoryTokenizer:
    chord_names_path = CHECKPOINT_DIR / "chord_names_augmented.json"
    with chord_names_path.open() as handle:
        chord_names = json.load(handle)
    return HooktheoryTokenizer(chord_names=chord_names)


def _online_override_args() -> dict:
    tokenizer = _tokenizer()
    return {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }


def _offline_override_args() -> dict:
    tokenizer = _tokenizer()
    return {
        "EncoderDecoderTransformer.enc_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.dec_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.pad_value": tokenizer.pad_token,
    }


@lru_cache(maxsize=None)
def _torch_online_model(model_path: str):
    path = Path(model_path)
    if path.suffix == ".pth":
        base_model = _torch_online_model(_resolve_existing_path(ONLINE_MODEL_CANDIDATES))
        model = load_rl_model(str(path), base_model, compile=False)
    else:
        model = load_gen_model_from_state_dict(
            str(path),
            DecoderTransformer,
            compile=False,
            override_args=_online_override_args(),
        )
    model.eval()
    return model


@lru_cache(maxsize=None)
def _mlx_online_model(model_path: str):
    path = Path(model_path)
    if path.suffix == ".pth":
        return MLXDecoderTransformer.from_rl_checkpoint(
            str(path),
            _tokenizer(),
            override_args=_online_override_args(),
        )
    return MLXDecoderTransformer.from_checkpoint(
        str(path),
        _tokenizer(),
        override_args=_online_override_args(),
    )


@lru_cache(maxsize=1)
def _torch_offline_model():
    model = load_gen_model_from_state_dict(
        _resolve_existing_path(OFFLINE_MODEL_CANDIDATES),
        EncoderDecoderTransformer,
        compile=False,
        override_args=_offline_override_args(),
    )
    model.eval()
    return model


@lru_cache(maxsize=1)
def _mlx_offline_model():
    return MLXEncoderDecoderTransformer.from_checkpoint(
        _resolve_existing_path(OFFLINE_MODEL_CANDIDATES),
        _tokenizer(),
        override_args=_offline_override_args(),
    )


def _online_sequence() -> torch.Tensor:
    tokenizer = _tokenizer()
    chord_name = tokenizer.chord_names[0]
    tokens = [
        tokenizer.bos_token,
        tokenizer.name_to_id[f"CHORD_ON_{chord_name}"],
        tokenizer.name_to_id["NOTE_ON_60"],
        tokenizer.name_to_id[f"CHORD_{chord_name}"],
        tokenizer.name_to_id["NOTE_60"],
        tokenizer.silence_token,
    ]
    return torch.tensor([tokens], dtype=torch.long)


def _offline_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = _tokenizer()
    chord_name = tokenizer.chord_names[0]
    enc_tokens = [
        tokenizer.bos_token,
        tokenizer.name_to_id["NOTE_ON_60"],
        tokenizer.name_to_id["NOTE_60"],
        tokenizer.silence_token,
        tokenizer.name_to_id["NOTE_ON_64"],
        tokenizer.name_to_id["NOTE_64"],
    ]
    dec_tokens = [
        tokenizer.bos_token,
        tokenizer.name_to_id[f"CHORD_ON_{chord_name}"],
        tokenizer.name_to_id[f"CHORD_{chord_name}"],
        tokenizer.silence_token,
    ]
    return (
        torch.tensor([enc_tokens], dtype=torch.long),
        torch.tensor([dec_tokens], dtype=torch.long),
    )


@unittest.skipUnless(MLX_AVAILABLE, "MLX is not installed")
class TestMLXRealJamParity(unittest.TestCase):
    def test_server_online_models_forward_parity(self):
        sequence = _online_sequence()

        for model_name, candidates in MODEL_PATH_CANDIDATES.items():
            with self.subTest(model=model_name):
                model_path = _resolve_existing_path(candidates)
                torch_model = _torch_online_model(model_path)
                mlx_model = _mlx_online_model(model_path)

                with torch.no_grad():
                    torch_logits = torch_model(sequence)
                mlx_logits = mlx_model(sequence)

                torch.testing.assert_close(
                    mlx_logits,
                    torch_logits,
                    rtol=1e-3,
                    atol=1e-3,
                )

    def test_online_generate_greedy_parity(self):
        tokenizer = _tokenizer()
        torch_model = _torch_online_model(
            _resolve_existing_path(ONLINE_MODEL_CANDIDATES)
        )
        mlx_model = _mlx_online_model(
            _resolve_existing_path(ONLINE_MODEL_CANDIDATES)
        )
        prompt = torch.tensor([[tokenizer.bos_token]], dtype=torch.long)
        filter_fn = partial(
            filter_invalid_tokens_generate_unconditional,
            model_part="chord",
            tokenizer=tokenizer,
            filter_opposite_part=True,
        )

        with torch.no_grad():
            torch_generated = torch_model.generate(
                prompts=prompt,
                seq_len=8,
                temperature=0.0,
                cache_kv=True,
                filter_logits_fn=filter_fn,
                filter_kwargs={},
            )

        mlx_generated = mlx_model.generate(
            prompts=prompt,
            seq_len=8,
            temperature=0.0,
            cache_kv=True,
            filter_logits_fn=filter_fn,
            filter_kwargs={},
        )

        self.assertTrue(torch.equal(mlx_generated, torch_generated))

    def test_offline_forward_parity(self):
        enc_tokens, dec_tokens = _offline_inputs()
        torch_model = _torch_offline_model()
        mlx_model = _mlx_offline_model()

        with torch.no_grad():
            torch_logits = torch_model(enc_tokens, dec_tokens)
        mlx_logits = mlx_model(enc_tokens, dec_tokens)

        torch.testing.assert_close(
            mlx_logits,
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_offline_generate_greedy_parity(self):
        tokenizer = _tokenizer()
        enc_tokens, _ = _offline_inputs()
        prompt = torch.tensor([[tokenizer.bos_token]], dtype=torch.long)
        torch_model = _torch_offline_model()
        mlx_model = _mlx_offline_model()
        filter_fn = partial(
            filter_invalid_tokens_generate_single_part,
            model_part="chord",
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            torch_generated = torch_model.generate(
                seq_in=enc_tokens,
                seq_out_start=prompt,
                seq_len=8,
                temperature=0.0,
                cache_kv=True,
                filter_logits_fn=filter_fn,
                filter_kwargs={},
            )

        mlx_generated = mlx_model.generate(
            seq_in=enc_tokens,
            seq_out_start=prompt,
            seq_len=8,
            temperature=0.0,
            cache_kv=True,
            filter_logits_fn=filter_fn,
            filter_kwargs={},
        )

        self.assertTrue(torch.equal(mlx_generated, torch_generated))
