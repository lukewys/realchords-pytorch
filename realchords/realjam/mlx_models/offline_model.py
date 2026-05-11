"""MLX encoder-decoder wrapper matching ReaLJam EncoderDecoderTransformer."""

from __future__ import annotations

from pathlib import Path

import argbind
import mlx.core as mx
import mlx.nn as nn
import torch
from torch.nn import functional as F

from realchords.model.sampling import (
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    top_k,
    validate_filter_fn_kwargs,
)
from realchords.nn.transformers import DEFAULT_KWARGS
from realchords.realjam.mlx_models.decoder import (
    MLXCrossAttnDecoderCore,
    MLXEncoderCore,
    _torch_to_mx,
    convert_cross_attn_decoder_weights,
    convert_encoder_weights,
    load_mlx_weights,
)
from realchords.realjam.mlx_models.online_model import (
    _load_raw_state_dict,
    _mx_to_torch,
)


def _enc_dec_config_from_args(args: dict, tokenizer) -> dict:
    attention_layer_configs = dict(DEFAULT_KWARGS)
    attention_layer_configs.update(
        args.get("EncoderDecoderTransformer.attention_layer_configs", {})
    )

    return {
        "enc_dim": args.get("EncoderDecoderTransformer.enc_dim", 512),
        "dec_dim": args.get("EncoderDecoderTransformer.dec_dim", 512),
        "enc_depth": args.get("EncoderDecoderTransformer.enc_depth", 6),
        "dec_depth": args.get("EncoderDecoderTransformer.dec_depth", 6),
        "enc_heads": args.get("EncoderDecoderTransformer.enc_heads", 8),
        "dec_heads": args.get("EncoderDecoderTransformer.dec_heads", 8),
        "enc_num_tokens": args.get(
            "EncoderDecoderTransformer.enc_num_tokens", tokenizer.num_tokens
        ),
        "dec_num_tokens": args.get(
            "EncoderDecoderTransformer.dec_num_tokens", tokenizer.num_tokens
        ),
        "enc_max_seq_len": args.get(
            "EncoderDecoderTransformer.enc_max_seq_len", 512
        ),
        "dec_max_seq_len": args.get(
            "EncoderDecoderTransformer.dec_max_seq_len", 512
        ),
        "pad_value": args.get(
            "EncoderDecoderTransformer.pad_value", tokenizer.pad_token
        ),
        "attention_layer_configs": attention_layer_configs,
    }


class MLXEncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        enc_dim: int = 512,
        dec_dim: int = 512,
        enc_depth: int = 6,
        dec_depth: int = 6,
        enc_heads: int = 8,
        dec_heads: int = 8,
        enc_num_tokens: int = 1024,
        dec_num_tokens: int = 1024,
        enc_max_seq_len: int = 512,
        dec_max_seq_len: int = 512,
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
                "MLX encoder-decoder requires use_simple_rmsnorm=True"
            )
        if not attention_layer_configs.get("attn_qk_norm", False):
            raise ValueError("MLX encoder-decoder requires attn_qk_norm=True")
        if not attention_layer_configs.get("ff_glu", False):
            raise ValueError("MLX encoder-decoder requires ff_glu=True")
        if not attention_layer_configs.get("ff_swish", False):
            raise ValueError("MLX encoder-decoder requires ff_swish=True")

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_heads = enc_heads
        self.dec_heads = dec_heads
        self.enc_num_tokens = enc_num_tokens
        self.dec_num_tokens = dec_num_tokens
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        self.pad_value = pad_value

        self.encoder_token_emb = nn.Embedding(enc_num_tokens, enc_dim)
        self.decoder_token_emb = nn.Embedding(dec_num_tokens, dec_dim)
        self.encoder = MLXEncoderCore(
            dim=enc_dim,
            depth=enc_depth,
            heads=enc_heads,
            kv_heads=enc_heads,
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
        self.decoder = MLXCrossAttnDecoderCore(
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            kv_heads=dec_heads,
            num_tokens=dec_num_tokens,
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

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        tokenizer,
        args: dict,
    ) -> "MLXEncoderDecoderTransformer":
        model = cls(**_enc_dec_config_from_args(args, tokenizer))

        weights = {
            "encoder_token_emb.weight": _torch_to_mx(
                state_dict["encoder.token_emb.emb.weight"]
            ),
            "decoder_token_emb.weight": _torch_to_mx(
                state_dict["decoder.decoder.token_emb.emb.weight"]
            ),
        }

        encoder_inner_state = {
            key[len("encoder.") :]: value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        decoder_inner_state = {
            key[len("decoder.decoder.") :]: value
            for key, value in state_dict.items()
            if key.startswith("decoder.decoder.")
        }

        weights.update(
            {
                f"encoder.{key}": value
                for key, value in convert_encoder_weights(
                    encoder_inner_state
                ).items()
            }
        )
        weights.update(
            {
                f"decoder.{key}": value
                for key, value in convert_cross_attn_decoder_weights(
                    decoder_inner_state
                ).items()
            }
        )
        load_mlx_weights(model, weights)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer,
        override_args: dict | None = None,
    ) -> "MLXEncoderDecoderTransformer":
        args = argbind.load_args(Path(checkpoint_path).parent / "args.yml")
        if override_args:
            args.update(override_args)
        state_dict = _load_raw_state_dict(checkpoint_path)
        return cls.from_state_dict(state_dict, tokenizer, args)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def _encode_mx(self, x_enc: torch.Tensor) -> mx.array:
        tokens = _torch_to_mx(x_enc).astype(mx.int32)
        embeds = self.encoder_token_emb(tokens)
        return self.encoder(embeds)

    def _decode_mx(
        self,
        x_dec: torch.Tensor,
        encoder_out: mx.array | None,
        cache=None,
    ) -> mx.array:
        tokens = _torch_to_mx(x_dec).astype(mx.int32)
        embeds = self.decoder_token_emb(tokens)
        return self.decoder(embeds, encoder_out=encoder_out, cache=cache)

    def __call__(
        self,
        x_enc,
        x_dec,
        enc_mask=None,
        dec_mask=None,
        return_attn_z_loss=False,
    ):
        if enc_mask is not None or dec_mask is not None:
            raise NotImplementedError(
                "MLX encoder-decoder does not support masks"
            )

        encodings = self._encode_mx(x_enc)
        logits = self._decode_mx(x_dec, encoder_out=encodings, cache=None)
        logits = _mx_to_torch(logits, device=x_dec.device)
        if return_attn_z_loss:
            return logits, torch.zeros(
                (), device=x_dec.device, dtype=logits.dtype
            )
        return logits

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_out_start,
        seq_len,
        mask=None,
        attn_mask=None,
        filter_logits_fn: str | callable | list = top_k,
        filter_kwargs: dict | list = dict(),
        temperature: float = 1.0,
        cache_kv: bool = True,
        eos_token=None,
        **kwargs,
    ):
        del attn_mask, kwargs

        if mask is not None:
            raise NotImplementedError(
                "MLX encoder-decoder does not support masks"
            )

        is_list = validate_filter_fn_kwargs(filter_logits_fn, filter_kwargs)
        if is_list:
            filter_logits_fn = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_kwargs = {}
        if isinstance(filter_logits_fn, str):
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        out = seq_out_start.clone()
        prompt_len = out.shape[-1]
        encoder_out = self._encode_mx(seq_in[:, -self.enc_max_seq_len :])

        cache = self.decoder.make_cache() if cache_kv else None
        self.decoder.precompute_cross_attn_kv(encoder_out)
        try:
            if cache_kv:
                last_logits = _mx_to_torch(
                    self._decode_mx(
                        out[:, -self.dec_max_seq_len :],
                        encoder_out=None,
                        cache=cache,
                    )[:, -1],
                    device=out.device,
                )

            for curr_sample_step in range(seq_len):
                if cache_kv:
                    logits = last_logits
                else:
                    logits = _mx_to_torch(
                        self._decode_mx(
                            out[:, -self.dec_max_seq_len :],
                            encoder_out=encoder_out,
                            cache=None,
                        )[:, -1],
                        device=out.device,
                    )

                if temperature == 0.0:
                    sample = logits.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits = filter_logits_fn(
                        logits,
                        curr_sample_step=curr_sample_step,
                        curr_sample_length=out.shape[-1],
                        curr_sequence=out,
                        **filter_kwargs,
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                if eos_token is not None:
                    is_eos_tokens = out == eos_token
                    if is_eos_tokens.any(dim=-1).all():
                        break

                if cache_kv and curr_sample_step < seq_len - 1:
                    last_logits = _mx_to_torch(
                        self._decode_mx(sample, encoder_out=None, cache=cache)[
                            :, -1
                        ],
                        device=out.device,
                    )
        finally:
            self.decoder.clear_cross_attn_kv()

        if eos_token is not None:
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            eos_mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(eos_mask, self.pad_value)

        return out[:, prompt_len:]
