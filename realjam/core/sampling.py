"""Sampling functions."""

import torch
import torch.nn.functional as F
from math import ceil

from typing import Callable, List, Optional

# The following functions are modified from x_transformers.
# We just added the kwargs to the functions to make them compatible with
#   other inference-time inputs.


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# nucleus


def top_p(logits, thres=0.9, **kwargs):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(
        sorted_indices_to_remove, (1, -1), value=False
    )

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk


def top_k(logits, frac_num_tokens=0.1, k=None, **kwargs):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


# top_a


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02, **kwargs):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float("-inf"), logits)


# min_p
# https://arxiv.org/abs/2407.01082


def min_p(logits, min_p=0.1, **kwargs):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


def filter_special_token(logits, pad_id=0, bos_id=1, eos_id=2, **kwargs):
    mask = torch.ones_like(logits)
    mask[:, pad_id] = 0
    mask[:, bos_id] = 0
    mask[:, eos_id] = 0
    return torch.where(mask == 0, float("-inf"), logits)


def filter_invalid_tokens(
    logits,
    model_part,
    previous_generated_token,
    tokenizer,
    mask_special_tokens=True,
    **kwargs,
):
    mask = torch.ones_like(logits)
    # mask special tokens
    if mask_special_tokens:
        special_token_range = tokenizer.special_token_range
        mask[:, special_token_range[0] : special_token_range[1] + 1] = 0

    # mask other part
    if model_part == "chord":
        note_token_range = tokenizer.note_token_range
        mask[:, note_token_range[0] : note_token_range[1] + 1] = 0
    elif model_part == "melody":
        chord_token_range = tokenizer.chord_token_range
        mask[:, chord_token_range[0] : chord_token_range[1] + 1] = 0
    else:
        raise ValueError(f"Invalid model part: {model_part}")

    if previous_generated_token is not None:
        for i in range(len(previous_generated_token)):
            prev_token = previous_generated_token[i].item()
            if model_part == "chord":
                if tokenizer.is_chord_on(prev_token):
                    chord_hold_id = tokenizer.chord_on_id_to_chord_id(
                        prev_token
                    )
                    chord_hold_range = tokenizer.chord_hold_token_range
                    mask[i, chord_hold_range[0] : chord_hold_range[1] + 1] = 0
                    mask[i, chord_hold_id] = 1
                elif tokenizer.is_chord_hold(prev_token):
                    chord_hold_id = prev_token
                    chord_hold_range = tokenizer.chord_hold_token_range
                    mask[i, chord_hold_range[0] : chord_hold_range[1] + 1] = 0
                    mask[i, chord_hold_id] = 1
                elif tokenizer.is_silence(prev_token):
                    pass
                else:
                    raise ValueError(f"Invalid chord token: {prev_token}")
            elif model_part == "melody":
                if tokenizer.is_note_on(prev_token):
                    note_hold_id = tokenizer.note_on_id_to_note_id(prev_token)
                    note_hold_range = tokenizer.note_hold_token_range
                    mask[i, note_hold_range[0] : note_hold_range[1] + 1] = 0
                    mask[i, note_hold_id] = 1
                elif tokenizer.is_note_hold(prev_token):
                    note_hold_id = prev_token
                    note_hold_range = tokenizer.note_hold_token_range
                    mask[i, note_hold_range[0] : note_hold_range[1] + 1] = 0
                    mask[i, note_hold_id] = 1
                elif tokenizer.is_silence(prev_token):
                    pass
                else:
                    raise ValueError(f"Invalid melody token: {prev_token}")
    else:
        if model_part == "chord":
            chord_hold_token_range = tokenizer.chord_hold_token_range
            mask[
                :, chord_hold_token_range[0] : chord_hold_token_range[1] + 1
            ] = 0
        elif model_part == "melody":
            note_hold_token_range = tokenizer.note_hold_token_range
            mask[:, note_hold_token_range[0] : note_hold_token_range[1] + 1] = 0

    return torch.where(mask == 0, float("-inf"), logits)


def filter_invalid_tokens_generate_online(
    logits,
    curr_sequence,
    curr_sample_step,
    model_part,
    tokenizer,
    mask_special_tokens=True,
    **kwargs,
):
    # previous generated token: 2 steps before
    # mask other chord hold tokens if the previous chord is a chord onset
    if curr_sequence.shape[1] > 3:  # at least 3 tokens (BOS + a pair of tokens)
        previous_generated_token = curr_sequence[:, -2]
    else:
        previous_generated_token = None

    logits = filter_invalid_tokens(
        logits,
        model_part,
        previous_generated_token,
        tokenizer,
        mask_special_tokens=mask_special_tokens,
    )

    return logits


def filter_invalid_tokens_generate_unconditional(
    logits,
    curr_sequence,
    curr_sample_step,
    model_part,
    tokenizer,
    mask_special_tokens=True,
    filter_opposite_part=True,
    **kwargs,
):
    opposite_model_part = "melody" if model_part == "chord" else "chord"
    if curr_sample_step % 2 == 0:
        curr_model_part = model_part
    else:
        curr_model_part = opposite_model_part
        if not filter_opposite_part:
            return logits

    if curr_sequence.shape[1] > 3:  # at least 3 tokens (BOS + a pair of tokens)
        previous_generated_token = curr_sequence[:, -2]
    else:
        previous_generated_token = None

    logits = filter_invalid_tokens(
        logits,
        curr_model_part,
        previous_generated_token,
        tokenizer,
        mask_special_tokens=mask_special_tokens,
    )
    return logits


def filter_invalid_tokens_generate_single_part(
    logits,
    curr_sequence,
    curr_sample_step,
    model_part,
    tokenizer,
    mask_special_tokens=True,
    **kwargs,
):
    # previous generated token: 1 steps before
    # mask other chord hold tokens if the previous chord is a chord onset
    if curr_sequence.shape[1] > 1:  # at least 2 tokens (BOS + any token)
        previous_generated_token = curr_sequence[:, -1]
    else:
        previous_generated_token = None

    logits = filter_invalid_tokens(
        logits,
        model_part,
        previous_generated_token,
        tokenizer,
        mask_special_tokens=mask_special_tokens,
    )

    return logits


def validate_filter_fn_kwargs(
    filter_logits_fn: str | List[str],
    filter_kwargs: dict | List[dict],
):
    lists = [
        isinstance(filter_logits_fn, list),
        isinstance(filter_kwargs, list),
    ]
    if any(lists) and not all(lists):
        raise ValueError(
            "filter_logits_fn and filter_kwargs must both be lists or "
            "both be single items."
        )
    if all(lists) and len(filter_logits_fn) != len(filter_kwargs):
        raise ValueError(
            "filter_logits_fn and filter_kwargs must have the same length."
        )
    is_list = all(lists)
    return is_list


# filter logits functions dict[str -> Callable]

FILTER_LOGITS_FN = dict(
    top_p=top_p,
    top_k=top_k,
    top_a=top_a,
    min_p=min_p,
    filter_special_token=filter_special_token,
)


class ComposeFilterFns:
    def __init__(
        self,
        filter_fns: List[str | Callable],
        filter_kwargs: Optional[List[dict]] = None,
    ):
        if isinstance(filter_fns[0], str):
            filter_fns = [FILTER_LOGITS_FN[fn] for fn in filter_fns]
        self.filter_fns = filter_fns

        self.filter_kwargs = default(filter_kwargs, [{}] * len(filter_fns))

    def __call__(self, logits, **kwargs):
        for filter_fn, filter_kwargs in zip(
            self.filter_fns, self.filter_kwargs
        ):
            fn_kwargs = {**filter_kwargs, **kwargs}
            logits = filter_fn(logits, **fn_kwargs)
        return logits
