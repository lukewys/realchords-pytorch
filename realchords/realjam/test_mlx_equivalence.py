#!/usr/bin/env python
"""Test numerical equivalence between PyTorch and MLX models.

Tests:
1. Forward pass equivalence (decoder-only / online model)
2. generate() equivalence (greedy, temperature=0)
3. generate_online() equivalence (greedy, temperature=0)
4. Forward pass equivalence (encoder-decoder / offline model)
5. generate() equivalence for encoder-decoder (greedy)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from lightning import seed_everything

from realchords.constants import REALJAM_CHECKPOINT_DIR
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.model.gen_model import (
    DecoderTransformer,
    EncoderDecoderTransformer,
)
from realchords.model.sampling import (
    filter_invalid_tokens_generate_online,
    filter_invalid_tokens_generate_unconditional,
)
from realchords.realjam.agent_interface import (
    OFFLINE_MODEL_CANDIDATES,
    ONLINE_MODEL_CANDIDATES,
    _resolve_existing_path,
)
from realchords.realjam.mlx_models import (
    MLXDecoderTransformer,
    MLXEncoderDecoderTransformer,
)
from realchords.utils.inference_utils import load_gen_model_from_state_dict


def load_tokenizer() -> HooktheoryTokenizer:
    chord_names_path = os.path.join(
        REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
    )
    with open(chord_names_path, "r") as f:
        chord_names = json.load(f)
    return HooktheoryTokenizer(chord_names=chord_names)


def get_sample_prompt(tokenizer: HooktheoryTokenizer) -> torch.Tensor:
    """Build a short sample prompt for testing."""
    sample = [
        tokenizer.bos_token,
        2099,
        187,
        748,
        59,
        748,
        59,
        748,
        59,
        748,
        59,
        748,
        59,
    ]
    return torch.tensor([sample], dtype=torch.long)


# =====================================================================
# Test 1: Decoder-only forward pass equivalence
# =====================================================================
def test_decoder_forward_equivalence():
    print("\n" + "=" * 70)
    print("TEST 1: Decoder-only forward pass equivalence (PyTorch vs MLX)")
    print("=" * 70)

    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(ONLINE_MODEL_CANDIDATES)

    override_args = {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }

    pt_model = load_gen_model_from_state_dict(
        model_path,
        DecoderTransformer,
        compile=False,
        override_args=override_args,
    )
    pt_model.eval()

    mlx_model = MLXDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    prompt = get_sample_prompt(tokenizer)

    with torch.no_grad():
        pt_logits = pt_model(prompt)

    mlx_logits = mlx_model(prompt)

    diff = (pt_logits - mlx_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Shape: PT={pt_logits.shape}, MLX={mlx_logits.shape}")
    print(f"  Max abs diff: {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")

    pt_argmax = pt_logits.argmax(dim=-1)
    mlx_argmax = mlx_logits.argmax(dim=-1)
    argmax_match = (pt_argmax == mlx_argmax).all().item()
    print(f"  Argmax matches: {argmax_match}")

    passed = max_diff < 0.01 and argmax_match
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# =====================================================================
# Test 2: Decoder-only generate (greedy) equivalence
# =====================================================================
def test_decoder_generate_equivalence():
    print("\n" + "=" * 70)
    print("TEST 2: Decoder-only generate (greedy) equivalence")
    print("=" * 70)

    seed_everything(42)
    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(ONLINE_MODEL_CANDIDATES)

    override_args = {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }

    pt_model = load_gen_model_from_state_dict(
        model_path,
        DecoderTransformer,
        compile=False,
        override_args=override_args,
    )
    pt_model.eval()

    mlx_model = MLXDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    prompt = get_sample_prompt(tokenizer)
    gen_len = 8

    filter_logits_fn = filter_invalid_tokens_generate_unconditional
    filter_kwargs = {
        "model_part": "chord",
        "tokenizer": tokenizer,
        "filter_opposite_part": True,
    }

    with torch.no_grad():
        pt_gen = pt_model.generate(
            prompts=prompt.clone(),
            seq_len=gen_len,
            temperature=0.0,
            cache_kv=True,
            filter_logits_fn=filter_logits_fn,
            filter_kwargs=filter_kwargs,
        )

    mlx_gen = mlx_model.generate(
        prompts=prompt.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=True,
        filter_logits_fn=filter_logits_fn,
        filter_kwargs=filter_kwargs,
    )

    print(f"  PT  generated: {pt_gen[0].tolist()}")
    print(f"  MLX generated: {mlx_gen[0].tolist()}")

    match = (pt_gen == mlx_gen).all().item()
    print(f"  Token match: {match}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    return match


# =====================================================================
# Test 3: Decoder-only generate_online (greedy) equivalence
# =====================================================================
def test_decoder_generate_online_equivalence():
    print("\n" + "=" * 70)
    print("TEST 3: Decoder-only generate_online (greedy) equivalence")
    print("=" * 70)

    seed_everything(42)
    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(ONLINE_MODEL_CANDIDATES)

    override_args = {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }

    pt_model = load_gen_model_from_state_dict(
        model_path,
        DecoderTransformer,
        compile=False,
        override_args=override_args,
    )
    pt_model.eval()

    mlx_model = MLXDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    prompt = get_sample_prompt(tokenizer)
    gen_len = 8
    conditions = torch.tensor([[748, 59, 748, 59]], dtype=torch.long)

    filter_logits_fn = filter_invalid_tokens_generate_online
    filter_kwargs = {
        "model_part": "melody",
        "tokenizer": tokenizer,
    }

    with torch.no_grad():
        pt_gen = pt_model.generate_online(
            prompts=prompt.clone(),
            conditions=conditions.clone(),
            seq_len=gen_len,
            temperature=0.0,
            cache_kv=True,
            filter_logits_fn=filter_logits_fn,
            filter_kwargs=filter_kwargs,
            reverse_condition_order=True,
        )

    mlx_gen = mlx_model.generate_online(
        prompts=prompt.clone(),
        conditions=conditions.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=True,
        filter_logits_fn=filter_logits_fn,
        filter_kwargs=filter_kwargs,
        reverse_condition_order=True,
    )

    print(f"  PT  generated: {pt_gen[0].tolist()}")
    print(f"  MLX generated: {mlx_gen[0].tolist()}")

    match = (pt_gen == mlx_gen).all().item()
    print(f"  Token match: {match}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    return match


# =====================================================================
# Test 4: Encoder-decoder forward pass equivalence
# =====================================================================
def test_enc_dec_forward_equivalence():
    print("\n" + "=" * 70)
    print("TEST 4: Encoder-decoder forward pass equivalence")
    print("=" * 70)

    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(OFFLINE_MODEL_CANDIDATES)

    override_args = {
        "EncoderDecoderTransformer.enc_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.dec_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.pad_value": tokenizer.pad_token,
    }

    pt_model = load_gen_model_from_state_dict(
        model_path,
        EncoderDecoderTransformer,
        compile=False,
        override_args=override_args,
    )
    pt_model.eval()

    mlx_model = MLXEncoderDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    enc_tokens = torch.tensor(
        [[tokenizer.bos_token, 187, 59, 59, 59, 59, tokenizer.eos_token]],
        dtype=torch.long,
    )
    dec_tokens = torch.tensor(
        [[tokenizer.bos_token, 2099, 748]],
        dtype=torch.long,
    )

    with torch.no_grad():
        pt_logits = pt_model(enc_tokens, dec_tokens)

    mlx_logits = mlx_model(enc_tokens, dec_tokens)

    diff = (pt_logits - mlx_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Shape: PT={pt_logits.shape}, MLX={mlx_logits.shape}")
    print(f"  Max abs diff: {max_diff:.8f}")
    print(f"  Mean abs diff: {mean_diff:.8f}")

    pt_argmax = pt_logits.argmax(dim=-1)
    mlx_argmax = mlx_logits.argmax(dim=-1)
    argmax_match = (pt_argmax == mlx_argmax).all().item()
    print(f"  Argmax matches: {argmax_match}")

    passed = max_diff < 0.01 and argmax_match
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# =====================================================================
# Test 5: Encoder-decoder generate (greedy) equivalence
# =====================================================================
def test_enc_dec_generate_equivalence():
    print("\n" + "=" * 70)
    print("TEST 5: Encoder-decoder generate (greedy) equivalence")
    print("=" * 70)

    seed_everything(42)
    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(OFFLINE_MODEL_CANDIDATES)

    override_args = {
        "EncoderDecoderTransformer.enc_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.dec_num_tokens": tokenizer.num_tokens,
        "EncoderDecoderTransformer.pad_value": tokenizer.pad_token,
    }

    pt_model = load_gen_model_from_state_dict(
        model_path,
        EncoderDecoderTransformer,
        compile=False,
        override_args=override_args,
    )
    pt_model.eval()

    mlx_model = MLXEncoderDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    enc_tokens = torch.tensor(
        [[tokenizer.bos_token, 187, 59, 59, 59, 59, tokenizer.eos_token]],
        dtype=torch.long,
    )
    dec_start = torch.tensor(
        [[tokenizer.bos_token]],
        dtype=torch.long,
    )
    gen_len = 8

    with torch.no_grad():
        pt_gen = pt_model.generate(
            seq_in=enc_tokens,
            seq_out_start=dec_start.clone(),
            seq_len=gen_len,
            temperature=0.0,
            cache_kv=True,
        )

    mlx_gen = mlx_model.generate(
        seq_in=enc_tokens,
        seq_out_start=dec_start.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=True,
    )

    print(f"  PT  generated: {pt_gen[0].tolist()}")
    print(f"  MLX generated: {mlx_gen[0].tolist()}")

    match = (pt_gen == mlx_gen).all().item()
    print(f"  Token match: {match}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    return match


# =====================================================================
# Test 6: MLX model generate_online cache consistency
# =====================================================================
def test_mlx_generate_online_cache_consistency():
    print("\n" + "=" * 70)
    print("TEST 6: MLX generate_online cache consistency")
    print("=" * 70)

    seed_everything(42)
    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(ONLINE_MODEL_CANDIDATES)

    override_args = {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }

    mlx_model = MLXDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    prompt = get_sample_prompt(tokenizer)
    gen_len = 6
    conditions = torch.tensor([[748, 59, 748]], dtype=torch.long)

    filter_logits_fn = filter_invalid_tokens_generate_online
    filter_kwargs = {
        "model_part": "melody",
        "tokenizer": tokenizer,
    }

    seed_everything(42)
    cached_gen = mlx_model.generate_online(
        prompts=prompt.clone(),
        conditions=conditions.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=True,
        filter_logits_fn=filter_logits_fn,
        filter_kwargs=filter_kwargs,
        reverse_condition_order=True,
    )

    seed_everything(42)
    uncached_gen = mlx_model.generate_online(
        prompts=prompt.clone(),
        conditions=conditions.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=False,
        filter_logits_fn=filter_logits_fn,
        filter_kwargs=filter_kwargs,
        reverse_condition_order=True,
    )

    print(f"  Cached:   {cached_gen[0].tolist()}")
    print(f"  Uncached: {uncached_gen[0].tolist()}")

    match = (cached_gen == uncached_gen).all().item()
    print(f"  Match: {match}")
    print(f"  RESULT: {'PASS' if match else 'FAIL'}")
    return match


# =====================================================================
# Test 7: MLX generate_online condition insertion order
# =====================================================================
def test_mlx_generate_online_condition_order():
    print("\n" + "=" * 70)
    print("TEST 7: MLX generate_online condition insertion order")
    print("=" * 70)

    seed_everything(42)
    tokenizer = load_tokenizer()
    model_path = _resolve_existing_path(ONLINE_MODEL_CANDIDATES)

    override_args = {
        "DecoderTransformer.num_tokens": tokenizer.num_tokens,
        "DecoderTransformer.pad_value": tokenizer.pad_token,
    }

    mlx_model = MLXDecoderTransformer.from_checkpoint(
        model_path, tokenizer, override_args=override_args
    )

    prompt = get_sample_prompt(tokenizer)
    gen_len = 6
    cond_val = 999
    conditions = torch.tensor(
        [[cond_val, cond_val, cond_val]], dtype=torch.long
    )

    gen = mlx_model.generate_online(
        prompts=prompt.clone(),
        conditions=conditions.clone(),
        seq_len=gen_len,
        temperature=0.0,
        cache_kv=True,
        reverse_condition_order=True,
    )

    gen_tokens = gen[0].tolist()
    print(f"  Generated (reverse=True): {gen_tokens}")

    even_match = all(gen_tokens[i] == cond_val for i in range(0, gen_len, 2))
    print(f"  Even positions have condition value: {even_match}")
    print(f"  RESULT: {'PASS' if even_match else 'FAIL'}")
    return even_match


# =====================================================================
# Test 8: onnx_kv curr_sample_length semantic correctness
# =====================================================================
def test_onnx_kv_curr_sequence_fix():
    print("\n" + "=" * 70)
    print("TEST 8: onnx_kv _build_curr_sequence fix verification")
    print("=" * 70)

    from realchords.realjam.onnx_kv import _build_curr_sequence

    prompt_np = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

    seq0 = _build_curr_sequence(prompt_np, [])
    assert seq0.shape == (1, 5), f"Expected (1,5), got {seq0.shape}"
    assert (
        seq0.numpy() == prompt_np
    ).all(), "Empty generated should return prompt"

    gen1 = np.array([[10]], dtype=np.int64)
    gen2 = np.array([[20]], dtype=np.int64)
    seq2 = _build_curr_sequence(prompt_np, [gen1, gen2])
    assert seq2.shape == (1, 7), f"Expected (1,7), got {seq2.shape}"
    expected = np.array([[1, 2, 3, 4, 5, 10, 20]], dtype=np.int64)
    assert (
        seq2.numpy() == expected
    ).all(), f"Expected {expected}, got {seq2.numpy()}"

    print("  _build_curr_sequence correctly includes prompt+generated")
    print("  RESULT: PASS")
    return True


# =====================================================================
# Main
# =====================================================================
def main():
    results = {}

    results["test_onnx_kv_curr_sequence_fix"] = test_onnx_kv_curr_sequence_fix()
    results["test_decoder_forward"] = test_decoder_forward_equivalence()
    results["test_decoder_generate"] = test_decoder_generate_equivalence()
    results["test_decoder_generate_online"] = (
        test_decoder_generate_online_equivalence()
    )
    results["test_enc_dec_forward"] = test_enc_dec_forward_equivalence()
    results["test_enc_dec_generate"] = test_enc_dec_generate_equivalence()
    results["test_mlx_cache_consistency"] = (
        test_mlx_generate_online_cache_consistency()
    )
    results["test_condition_order"] = test_mlx_generate_online_condition_order()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
