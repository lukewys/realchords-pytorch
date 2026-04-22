"""Generate function for MARL interaction."""

import torch
from torch import Tensor
from torch.nn import functional as F
from einops import pack, unpack
from typing import Callable, List

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    exists,
    join,
)

from realchords.model.sampling import (
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    validate_filter_fn_kwargs,
    top_k,
)
from realchords.utils.sequence_utils import sequences_order_to_counterpart


def sample_from_logits(
    logits,
    greedy,
    temperature,
    filter_logits_fn,
    curr_sequence,
    curr_sample_step,
    filter_kwargs,
):
    if greedy:
        sample = logits.argmax(dim=-1, keepdim=True)
    else:
        filtered_logits = filter_logits_fn(
            logits,
            curr_sequence=curr_sequence,
            curr_sample_step=curr_sample_step,
            **filter_kwargs,
        )
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
    return sample


@torch.no_grad()
@eval_decorator
def generate_marl(
    model_1,
    model_2,
    prompts,
    seq_len,
    eos_token=None,
    temperature_1=1.0,
    temperature_2=1.0,
    filter_logits_fn_1: str | Callable | List[str | Callable] = top_k,
    filter_logits_fn_2: str | Callable | List[str | Callable] = top_k,
    filter_kwargs_1: dict | List[dict] = dict(),
    filter_kwargs_2: dict | List[dict] = dict(),
    cache_kv=True,
    debug=False,
    **kwargs,
):
    """Generate tokens for multi-agent reinforcement learning (MARL) with two models."""

    filter_fns_is_list_1 = validate_filter_fn_kwargs(
        filter_logits_fn_1, filter_kwargs_1
    )
    if filter_fns_is_list_1:
        filter_fns_1 = ComposeFilterFns(filter_logits_fn_1, filter_kwargs_1)
        filter_logits_fn_1 = filter_fns_1
        filter_kwargs_1 = dict()
    elif isinstance(filter_logits_fn_1, str):
        assert (
            filter_logits_fn_1 in FILTER_LOGITS_FN
        ), f"only {join(FILTER_LOGITS_FN.keys())} are available"

        filter_logits_fn_1 = FILTER_LOGITS_FN[filter_logits_fn_1]

    filter_fns_is_list_2 = validate_filter_fn_kwargs(
        filter_logits_fn_2, filter_kwargs_2
    )
    if filter_fns_is_list_2:
        filter_fns_2 = ComposeFilterFns(filter_logits_fn_2, filter_kwargs_2)
        filter_logits_fn_2 = filter_fns_2
        filter_kwargs_2 = dict()
    elif isinstance(filter_logits_fn_2, str):
        assert (
            filter_logits_fn_2 in FILTER_LOGITS_FN
        ), f"only {join(FILTER_LOGITS_FN.keys())} are available"

        filter_logits_fn_2 = FILTER_LOGITS_FN[filter_logits_fn_2]

    if model_1.max_seq_len != model_2.max_seq_len:
        raise ValueError(
            "model_1 and model_2 must have the same max sequence length"
        )

    max_seq_len, greedy = (
        model_1.max_seq_len,
        temperature_1 == 0.0,
    )

    if seq_len % 2 != 0:
        raise ValueError("seq_len must be even")

    prompts, ps = pack([prompts], "* n")

    b, t = prompts.shape

    out_1 = prompts
    out_2 = sequences_order_to_counterpart(prompts)

    cache_1 = None
    cache_2 = None

    sample_1_prev = None
    sample_2_prev = None

    for step in range(seq_len):
        max_len_exceeded = out_1.shape[-1] > max_seq_len

        if max_len_exceeded:
            raise ValueError(
                "the network cannot use cached key values when decoding "
                "outside the max sequence length."
            )

        if step % 2 == 0:
            if sample_2_prev is not None:
                _, new_cache_1 = model_1(
                    out_1,
                    return_intermediates=True,
                    cache=cache_1,
                    **kwargs,
                )

                if cache_kv and model_1.net.can_cache_kv:
                    cache_1 = new_cache_1

                out_1 = torch.cat((out_1, sample_2_prev), dim=-1)

                if debug:
                    print(
                        f"step {step}, model 1 context appended: {sample_2_prev}"
                    )

            logits_1, new_cache_1 = model_1(
                out_1,
                return_intermediates=True,
                cache=cache_1,
                **kwargs,
            )

            if cache_kv and model_1.net.can_cache_kv:
                cache_1 = new_cache_1

            logits_1 = logits_1[:, -1]

            sample_1 = sample_from_logits(
                logits_1,
                greedy,
                temperature_1,
                filter_logits_fn_1,
                curr_sequence=out_1,
                curr_sample_step=step,
                filter_kwargs=filter_kwargs_1,
            )
            out_1 = torch.cat((out_1, sample_1), dim=-1)

            if debug:
                print(f"step {step}, model 1 sample: {sample_1}")

            sample_1_prev = sample_1

        else:
            logits_2, new_cache_2 = model_2(
                out_2,
                return_intermediates=True,
                cache=cache_2,
                **kwargs,
            )

            if cache_kv and model_2.net.can_cache_kv:
                cache_2 = new_cache_2

            logits_2 = logits_2[:, -1]

            sample_2 = sample_from_logits(
                logits_2,
                greedy,
                temperature_2,
                filter_logits_fn_2,
                curr_sequence=out_2,
                curr_sample_step=step,
                filter_kwargs=filter_kwargs_2,
            )
            out_2 = torch.cat((out_2, sample_2), dim=-1)

            if debug:
                print(f"step {step}, model 2 sample: {sample_2}")

            if sample_1_prev is not None:
                _, new_cache_2 = model_2(
                    out_2,
                    return_intermediates=True,
                    cache=cache_2,
                    **kwargs,
                )

                if cache_kv and model_2.net.can_cache_kv:
                    cache_2 = new_cache_2

                out_2 = torch.cat((out_2, sample_1_prev), dim=-1)

                if debug:
                    print(
                        f"step {step}, model 2 context appended: {sample_1_prev}"
                    )

            sample_2_prev = sample_2

        if not exists(eos_token):
            continue

        is_eos_tokens_1 = out_1 == eos_token
        is_eos_tokens_2 = out_2 == eos_token

        if (
            is_eos_tokens_1.any(dim=-1).all()
            and is_eos_tokens_2.any(dim=-1).all()
        ):
            break

    out_1 = torch.cat((out_1, sample_2_prev), dim=-1)

    if exists(eos_token):
        shifted_is_eos_tokens_1 = F.pad(is_eos_tokens_1, (1, -1))
        mask_1 = shifted_is_eos_tokens_1.float().cumsum(dim=-1) >= 1
        out_1 = out_1.masked_fill(mask_1, model_1.pad_value)
        shifted_is_eos_tokens_2 = F.pad(is_eos_tokens_2, (1, -1))
        mask_2 = shifted_is_eos_tokens_2.float().cumsum(dim=-1) >= 1
        out_2 = out_2.masked_fill(mask_2, model_2.pad_value)

    out_1 = out_1[:, t:]
    out_2 = out_2[:, t:]

    (out_1,) = unpack(out_1, ps, "* n")
    (out_2,) = unpack(out_2, ps, "* n")

    return out_1, out_2
