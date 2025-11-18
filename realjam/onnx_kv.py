"""
ONNX Runtime-based autoregressive generation for online-KV models.
Exposed entry point: ``generate_tokens``.
"""

from __future__ import annotations
from typing import Tuple, Callable, Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F


# ------------------------- helpers ----------------------------------
def _top_k_np(logits: np.ndarray, k: int | None) -> np.ndarray:
    """Keep only the top-k values (per row).  No-op if k is None/≤0."""
    if k is None or k <= 0:
        return logits
    idx = np.argsort(-logits, axis=-1)
    kth = np.take_along_axis(logits, idx[:, k - 1 : k], axis=-1)
    return np.where(logits < kth, -1e10, logits)


def _build_packed_prompt(
    prompt_ids: torch.Tensor,
    max_len: int,
) -> np.ndarray:
    """Create (B, max_len, 2) packed tensor expected by ONNX graph."""
    if torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids.cpu().numpy()

    b, t = prompt_ids.shape
    packed = np.zeros((b, max_len, 2), dtype=np.int64)
    packed[:, -t:, 0] = prompt_ids
    packed[:, -t:, 1] = 1
    return packed


# ------------------------- main API ---------------------------------
def generate_tokens(
    session_init: ort.InferenceSession,
    session_step: ort.InferenceSession,
    prompt_ids: torch.Tensor,
    n_new: int = 32,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    max_len: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    temperature : float
        Softmax temperature (>0). Must match PyTorch path for parity.
    top_k : int | None
        Top-k to apply in *numpy* mode. Ignored if `filter_logits_fn`
        already performs top-k in *torch* mode.
    sampling : {"torch","numpy"}
        `"torch"`  – converts logits to torch, runs
                     `filter_logits_fn → softmax → multinomial`
        `"numpy"`  – pure numpy sampling (top_k + softmax + np.choice)
    filter_logits_fn / filter_kwargs
        Same callable + kwargs your PyTorch code passes.

    Returns
    -------
    new_tokens : (B, n_new)  int64
    all_logits : (B, n_new, V)  raw pre-softmax logits for every step
    """

    eps = 1e-8
    bsz = prompt_ids.shape[0]
    filter_kwargs = filter_kwargs or {}

    # ---- first frame ------------------------------------------------
    packed = _build_packed_prompt(prompt_ids, max_len)
    logits, *_, kv0, kv1, _ = session_init.run(None, {"gen_inputs": packed})

    generated: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for step in range(n_new):
        step_logits = logits[:, -1, :]  # (B, V)
        all_logits.append(step_logits)

        torch_logits = torch.from_numpy(step_logits.astype(np.float32))

        # apply caller-supplied filter (e.g. top_k helper)
        if filter_logits_fn is not None:
            torch_logits = filter_logits_fn(
                torch_logits,
                curr_sample_step=step,
                curr_sample_length=len(generated),
                curr_sequence=(
                    torch.from_numpy(np.concatenate(generated, axis=1))
                    if generated
                    else None
                ),
                **filter_kwargs,
            )

        probs_t = F.softmax(torch_logits / max(temperature, eps), dim=-1)
        next_tok = torch.multinomial(probs_t, 1).cpu().numpy()

        generated.append(next_tok)

        if step == n_new - 1:
            break

        # ---- slide packed window & call step graph -------------------
        packed = np.roll(packed, -1, axis=1)
        packed[:, -1, 0] = next_tok.squeeze(-1)
        packed[:, -1, 1] = 1

        logits, *_, kv0, kv1, _ = session_step.run(
            None,
            {"gen_inputs": packed, "cached_kv0": kv0, "cached_kv1": kv1},
        )

    new_tokens_np = np.concatenate(generated, axis=1)  # (B, n_new)
    all_logits_np = np.stack(all_logits, axis=1)  # (B, n_new, V)
    return new_tokens_np, all_logits_np


def generate_tokens_online(
    session_init: ort.InferenceSession,
    session_step: ort.InferenceSession,
    prompt_ids: torch.Tensor,
    conditions: torch.Tensor,  # condition tokens to insert
    n_new: int = 32,
    *,
    temperature: float = 1.0,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    max_len: int = 512,
    reverse_condition_order: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate tokens with alternating token insertion from conditions.

    Parameters
    ----------
    session_init, session_step : onnxruntime.InferenceSession
        The ONNX inference sessions for initialization and step
    prompt_ids : torch.Tensor
        Input prompt tokens (B, T)
    conditions : torch.Tensor
        Condition tokens to insert (B, C)
    n_new : int
        Number of new tokens to generate (total, including condition insertions)
    temperature : float
        Sampling temperature
    filter_logits_fn, filter_kwargs
        Same callable + kwargs from PyTorch code
    max_len : int
        Maximum length for ONNX model context
    reverse_condition_order : bool
        If True, condition tokens are inserted at even positions

    Returns
    -------
    new_tokens : (B, n_new) int64
    all_logits : (B, n_new//2, V) raw pre-softmax logits for generated tokens
    """
    eps = 1e-8
    bsz = prompt_ids.shape[0]
    filter_kwargs = filter_kwargs or {}

    if torch.is_tensor(prompt_ids):
        prompt_np = prompt_ids.cpu().numpy()
    else:
        prompt_np = prompt_ids

    if torch.is_tensor(conditions):
        conditions_np = conditions.cpu().numpy()
    else:
        conditions_np = conditions

    # ---- first frame ------------------------------------------------
    packed = _build_packed_prompt(prompt_ids, max_len)
    logits, *_, kv0, kv1, _ = session_init.run(None, {"gen_inputs": packed})

    generated = []
    all_logits = []

    if reverse_condition_order:
        check_is_model_part = lambda x: x % 2 == 1
    else:
        check_is_model_part = lambda x: x % 2 == 0

    for step in range(n_new):
        if check_is_model_part(step):
            # Sample from model
            step_logits = logits[:, -1, :]  # (B, V)
            all_logits.append(step_logits)

            torch_logits = torch.from_numpy(step_logits.astype(np.float32))

            # Apply caller-supplied filter
            if filter_logits_fn is not None:
                torch_logits = filter_logits_fn(
                    torch_logits,
                    curr_sample_step=step,
                    curr_sample_length=len(generated),
                    curr_sequence=(
                        torch.from_numpy(np.concatenate(generated, axis=1))
                        if generated
                        else None
                    ),
                    **filter_kwargs,
                )

            probs_t = F.softmax(torch_logits / max(temperature, eps), dim=-1)
            next_tok = torch.multinomial(probs_t, 1).cpu().numpy()
            generated.append(next_tok)
        else:
            # Use condition token
            condition_idx = step // 2
            if condition_idx < conditions_np.shape[1]:
                next_tok = conditions_np[:, condition_idx : condition_idx + 1]
                generated.append(next_tok)
            else:
                # Run out of conditions, just sample
                step_logits = logits[:, -1, :]
                torch_logits = torch.from_numpy(step_logits.astype(np.float32))

                if filter_logits_fn is not None:
                    torch_logits = filter_logits_fn(
                        torch_logits,
                        curr_sample_step=step,
                        curr_sample_length=len(generated),
                        curr_sequence=(
                            torch.from_numpy(np.concatenate(generated, axis=1))
                            if generated
                            else None
                        ),
                        **filter_kwargs,
                    )

                probs_t = F.softmax(
                    torch_logits / max(temperature, eps), dim=-1
                )
                next_tok = torch.multinomial(probs_t, 1).cpu().numpy()
                generated.append(next_tok)

        if step == n_new - 1:
            break

        # ---- slide packed window & call step graph -------------------
        packed = np.roll(packed, -1, axis=1)
        packed[:, -1, 0] = next_tok.squeeze(-1)
        packed[:, -1, 1] = 1

        logits, *_, kv0, kv1, _ = session_step.run(
            None,
            {"gen_inputs": packed, "cached_kv0": kv0, "cached_kv1": kv1},
        )

    new_tokens_np = np.concatenate(generated, axis=1)  # (B, n_new)
    all_logits_np = (
        np.stack(all_logits, axis=1)
        if all_logits
        else np.zeros((bsz, 0, logits.shape[-1]))
    )
    return new_tokens_np, all_logits_np
