"""MLX decoder-only wrapper matching ReaLJam DecoderTransformer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import argbind
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from torch.nn import functional as F
from x_transformers.autoregressive_wrapper import align_right, exists, join

from realchords.model.sampling import (
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    top_k,
    validate_filter_fn_kwargs,
)
from realchords.nn.transformers import DEFAULT_KWARGS
from realchords.realjam.mlx_models.decoder import (
    MLXDecoderCore,
    _torch_to_mx,
    convert_decoder_weights,
    load_mlx_weights,
)


def _mx_to_torch(
    array: mx.array, device: torch.device | None = None
) -> torch.Tensor:
    mx.eval(array)
    tensor = torch.from_numpy(np.asarray(array))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _load_raw_state_dict(checkpoint_path: str) -> dict:
    state_dict = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}


def _decoder_config_from_args(args: dict, tokenizer) -> dict:
    attention_layer_configs = dict(DEFAULT_KWARGS)
    attention_layer_configs.update(
        args.get("DecoderTransformer.attention_layer_configs", {})
    )

    return {
        "dim": args.get("DecoderTransformer.dim", 512),
        "depth": args.get("DecoderTransformer.depth", 6),
        "heads": args.get("DecoderTransformer.heads", 8),
        "num_tokens": args.get(
            "DecoderTransformer.num_tokens", tokenizer.num_tokens
        ),
        "max_seq_len": args.get("DecoderTransformer.max_seq_len", 512),
        "pad_value": args.get(
            "DecoderTransformer.pad_value", tokenizer.pad_token
        ),
        "attention_layer_configs": attention_layer_configs,
    }


class MLXDecoderTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        pad_value: int = 0,
        attention_layer_configs: dict | None = None,
    ):
        super().__init__()
        attention_layer_configs = {
            **DEFAULT_KWARGS,
            **(attention_layer_configs or {}),
        }

        if not attention_layer_configs.get("use_simple_rmsnorm", False):
            raise ValueError(
                "MLX decoder only supports use_simple_rmsnorm=True"
            )
        if not attention_layer_configs.get("attn_qk_norm", False):
            raise ValueError("MLX decoder only supports attn_qk_norm=True")
        if not attention_layer_configs.get("ff_glu", False):
            raise ValueError("MLX decoder only supports ff_glu=True")
        if not attention_layer_configs.get("ff_swish", False):
            raise ValueError("MLX decoder only supports ff_swish=True")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.pad_value = pad_value

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.decoder = MLXDecoderCore(
            dim=dim,
            depth=depth,
            heads=heads,
            kv_heads=heads,
            num_tokens=num_tokens,
            rotary_pos_emb=attention_layer_configs.get("rotary_pos_emb", False),
            rotary_emb_dim=attention_layer_configs.get("rotary_emb_dim"),
            rel_pos_bias=attention_layer_configs.get("rel_pos_bias", False),
            rel_pos_num_buckets=attention_layer_configs.get(
                "rel_pos_num_buckets", 32
            ),
            rel_pos_max_distance=attention_layer_configs.get(
                "rel_pos_max_distance", 128
            ),
        )

    @property
    def net(self):
        return self

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        tokenizer,
        args: dict,
    ) -> "MLXDecoderTransformer":
        model = cls(**_decoder_config_from_args(args, tokenizer))

        weights = {
            "token_emb.weight": _torch_to_mx(
                state_dict["decoder.token_emb.emb.weight"]
            )
        }
        inner_state = {
            key[len("decoder.") :]: value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
        }
        converted = convert_decoder_weights(inner_state)
        weights.update(
            {f"decoder.{key}": value for key, value in converted.items()}
        )
        load_mlx_weights(model, weights)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer,
        override_args: dict | None = None,
    ) -> "MLXDecoderTransformer":
        args = argbind.load_args(Path(checkpoint_path).parent / "args.yml")
        if override_args:
            args.update(override_args)
        state_dict = _load_raw_state_dict(checkpoint_path)
        return cls.from_state_dict(state_dict, tokenizer, args)

    @classmethod
    def from_rl_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer,
        override_args: dict | None = None,
    ) -> "MLXDecoderTransformer":
        args = argbind.load_args(Path(checkpoint_path).parent / "args.yml")
        if override_args:
            args.update(override_args)

        raw_state = _load_raw_state_dict(checkpoint_path)
        state_dict = {}
        for key, value in raw_state.items():
            if key.startswith("model.module."):
                state_dict[key[len("model.module.") :]] = value
        return cls.from_state_dict(state_dict, tokenizer, args)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def _forward_mx(
        self,
        token_ids: torch.Tensor,
        cache=None,
    ) -> mx.array:
        token_ids_mx = _torch_to_mx(token_ids).astype(mx.int32)
        embeds = self.token_emb(token_ids_mx)
        return self.decoder(embeds, cache=cache)

    def __call__(self, x, mask=None, **kwargs):
        if mask is not None:
            raise NotImplementedError("MLX decoder does not support masks")
        if kwargs.get("context") is not None:
            raise NotImplementedError(
                "MLX decoder does not support cross-attend"
            )
        logits = self._forward_mx(x, cache=None)
        return _mx_to_torch(logits, device=x.device)

    def _prepare_filter(
        self,
        filter_logits_fn: str | Callable | List[str | Callable],
        filter_kwargs: dict | List[dict],
    ) -> tuple[Callable, dict]:
        is_list = validate_filter_fn_kwargs(filter_logits_fn, filter_kwargs)
        if is_list:
            filter_logits_fn = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_kwargs = {}

        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        return filter_logits_fn, filter_kwargs

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        out: torch.Tensor,
        curr_sample_step: int,
        temperature: float,
        filter_logits_fn: Callable,
        filter_kwargs: dict,
    ) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        filtered_logits = filter_logits_fn(
            logits,
            curr_sample_step=curr_sample_step,
            curr_sample_length=out.shape[-1],
            curr_sequence=out,
            **filter_kwargs,
        )
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        return torch.multinomial(probs, 1)

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: torch.Tensor | None = None,
        filter_logits_fn: str | Callable | List[str | Callable] = top_k,
        restrict_to_max_seq_len=True,
        filter_kwargs: dict | List[dict] = dict(),
        cache_kv=True,
        display_pbar=False,
        **kwargs,
    ):
        del display_pbar, kwargs

        if exists(prompt_lens):
            raise NotImplementedError(
                "MLX decoder does not support prompt_lens"
            )

        filter_logits_fn, filter_kwargs = self._prepare_filter(
            filter_logits_fn, filter_kwargs
        )

        if restrict_to_max_seq_len and prompts.shape[-1] > self.max_seq_len:
            prompts = prompts[:, -self.max_seq_len :]

        out = prompts.clone()
        prompt_len = out.shape[-1]
        cache = self.decoder.make_cache() if cache_kv else None

        if cache_kv:
            last_logits = _mx_to_torch(
                self._forward_mx(out, cache=cache)[:, -1],
                device=out.device,
            )

        for curr_sample_step in range(seq_len):
            if cache_kv:
                logits = last_logits
            else:
                window = (
                    out[:, -self.max_seq_len :]
                    if restrict_to_max_seq_len
                    else out
                )
                logits = _mx_to_torch(
                    self._forward_mx(window, cache=None)[:, -1],
                    device=out.device,
                )

            sample = self._sample_from_logits(
                logits,
                out,
                curr_sample_step,
                temperature,
                filter_logits_fn,
                filter_kwargs,
            )
            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    break

            if cache_kv and curr_sample_step < seq_len - 1:
                last_logits = _mx_to_torch(
                    self._forward_mx(sample, cache=cache)[:, -1],
                    device=out.device,
                )

        if exists(eos_token):
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        return out[:, prompt_len:]

    @torch.no_grad()
    def generate_online(
        self,
        prompts,
        conditions,
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: torch.Tensor | None = None,
        filter_logits_fn: str | Callable | List[str | Callable] = top_k,
        restrict_to_max_seq_len=True,
        filter_kwargs: dict | List[dict] = dict(),
        cache_kv=True,
        reverse_condition_order=False,
        **kwargs,
    ):
        del kwargs

        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            raise NotImplementedError(
                "MLX decoder does not support prompt_lens"
            )

        filter_logits_fn, filter_kwargs = self._prepare_filter(
            filter_logits_fn, filter_kwargs
        )

        if restrict_to_max_seq_len and prompts.shape[-1] > self.max_seq_len:
            prompts = prompts[:, -self.max_seq_len :]

        out = prompts.clone()
        prompt_len = out.shape[-1]
        cache = self.decoder.make_cache() if cache_kv else None

        if reverse_condition_order:
            is_model_step = lambda step: step % 2 == 1
        else:
            is_model_step = lambda step: step % 2 == 0

        if cache_kv:
            last_logits = _mx_to_torch(
                self._forward_mx(out, cache=cache)[:, -1],
                device=out.device,
            )

        for step in range(seq_len):
            if is_model_step(step):
                if cache_kv:
                    logits = last_logits
                else:
                    window = (
                        out[:, -self.max_seq_len :]
                        if restrict_to_max_seq_len
                        else out
                    )
                    logits = _mx_to_torch(
                        self._forward_mx(window, cache=None)[:, -1],
                        device=out.device,
                    )

                sample = self._sample_from_logits(
                    logits,
                    out,
                    step,
                    temperature,
                    filter_logits_fn,
                    filter_kwargs,
                )
            else:
                sample = conditions[:, step // 2].unsqueeze(-1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    break

            if cache_kv and step < seq_len - 1:
                last_logits = _mx_to_torch(
                    self._forward_mx(sample, cache=cache)[:, -1],
                    device=out.device,
                )

        if exists(eos_token):
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        return out[:, prompt_len:]
