"""Shared MLX transformer components for ReaLJam inference."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _torch_to_mx(tensor) -> mx.array:
    import torch

    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return mx.array(tensor.detach().cpu().numpy())
    return mx.array(tensor)


def _l2_normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
    return x / mx.maximum(norm, eps)


def _rotate_half(x: mx.array) -> mx.array:
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(shape)


def _apply_rotary_emb(t: mx.array, freqs: mx.array) -> mx.array:
    rot_dim = freqs.shape[-1]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = mx.expand_dims(freqs, axis=1)
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = t_rot * mx.cos(freqs) + _rotate_half(t_rot) * mx.sin(freqs)
    return mx.concatenate([t_rot, t_pass], axis=-1)


class MLXRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self._dim = dim
        inv_freq = 1.0 / (
            base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, positions: mx.array) -> mx.array:
        if positions.ndim == 1:
            positions = positions[None]
        freqs = positions[..., None] * self.inv_freq[None, None, :]
        freqs = mx.stack([freqs, freqs], axis=-1)
        return freqs.reshape(*positions.shape, self._dim)


class MLXRelativePositionBias(nn.Module):
    def __init__(
        self,
        scale: float,
        causal: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        heads: int = 8,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: mx.array,
        causal: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> mx.array:
        ret = mx.zeros_like(relative_position)
        n = -relative_position

        if not causal:
            num_buckets //= 2
            ret = ret + (n < 0).astype(mx.int32) * num_buckets
            n = mx.abs(n)
        else:
            n = mx.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        n_float = n.astype(mx.float32)
        safe_n = mx.maximum(n_float, float(max_exact))
        val_if_large = max_exact + (
            mx.log(safe_n / float(max_exact))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        val_if_large = mx.minimum(val_if_large, num_buckets - 1)

        return ret + mx.where(is_small, n, val_if_large)

    def __call__(self, query_len: int, key_len: int) -> mx.array:
        q_pos = mx.arange(key_len - query_len, key_len, dtype=mx.int32)
        k_pos = mx.arange(key_len, dtype=mx.int32)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        buckets = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(buckets)
        return values.transpose(2, 0, 1) * self.scale


class MLXSimpleRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)

    def __call__(self, x: mx.array) -> mx.array:
        return _l2_normalize(x) * self.scale


class MLXKVCache:
    def __init__(self, num_layers: int):
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers
        self.offset = 0

    def update(
        self,
        layer_idx: int,
        key: mx.array,
        value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = key
            self.values[layer_idx] = value
        else:
            self.keys[layer_idx] = mx.concatenate(
                [self.keys[layer_idx], key], axis=2
            )
            self.values[layer_idx] = mx.concatenate(
                [self.values[layer_idx], value], axis=2
            )
        return self.keys[layer_idx], self.values[layer_idx]

    def advance(self, n_tokens: int) -> None:
        self.offset += n_tokens

    def reset(self) -> None:
        num_layers = len(self.keys)
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.offset = 0


class MLXSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.qk_norm_scale = qk_norm_scale

        self.to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, dim, bias=False)

        self.qk_norm_q_scale = mx.ones((heads, 1, head_dim))
        self.qk_norm_k_scale = mx.ones((kv_heads, 1, head_dim))

    def __call__(
        self,
        x: mx.array,
        rotary_freqs: mx.array | None,
        rel_pos: MLXRelativePositionBias | None = None,
        cache: MLXKVCache | None = None,
        layer_idx: int = 0,
        causal: bool = True,
    ) -> mx.array:
        batch_size, query_len, _ = x.shape

        q = self.to_q(x).reshape(
            batch_size, query_len, self.heads, self.head_dim
        )
        k = self.to_k(x).reshape(
            batch_size, query_len, self.kv_heads, self.head_dim
        )
        v = self.to_v(x).reshape(
            batch_size, query_len, self.kv_heads, self.head_dim
        )

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = _l2_normalize(q) * self.qk_norm_q_scale
        k = _l2_normalize(k) * self.qk_norm_k_scale

        if rotary_freqs is not None:
            q = _apply_rotary_emb(q, rotary_freqs)
            k = _apply_rotary_emb(k, rotary_freqs)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            k = mx.tile(k, [1, repeats, 1, 1])
            v = mx.tile(v, [1, repeats, 1, 1])

        total_kv = k.shape[2]
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.qk_norm_scale

        if rel_pos is not None:
            scores = scores + rel_pos(query_len, total_kv)[None]

        if causal:
            q_pos = mx.arange(total_kv - query_len, total_kv, dtype=mx.int32)
            k_pos = mx.arange(total_kv, dtype=mx.int32)
            allowed = k_pos[None, :] <= q_pos[:, None]
            scores = mx.where(
                allowed[None, None],
                scores,
                mx.full(scores.shape, -1e9, dtype=scores.dtype),
            )

        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        out = mx.matmul(weights.astype(v.dtype), v)
        out = out.transpose(0, 2, 1, 3).reshape(
            batch_size, query_len, self.heads * self.head_dim
        )
        return self.to_out(out)


class MLXCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.qk_norm_scale = qk_norm_scale

        self.to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.to_out = nn.Linear(heads * head_dim, dim, bias=False)

        self.qk_norm_q_scale = mx.ones((heads, 1, head_dim))
        self.qk_norm_k_scale = mx.ones((kv_heads, 1, head_dim))

        self._cached_k: mx.array | None = None
        self._cached_v: mx.array | None = None

    def precompute_kv(self, encoder_out: mx.array) -> None:
        batch_size, seq_len, _ = encoder_out.shape
        k = self.to_k(encoder_out).reshape(
            batch_size, seq_len, self.kv_heads, self.head_dim
        )
        v = self.to_v(encoder_out).reshape(
            batch_size, seq_len, self.kv_heads, self.head_dim
        )
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        k = _l2_normalize(k) * self.qk_norm_k_scale

        if self.kv_heads != self.heads:
            repeats = self.heads // self.kv_heads
            k = mx.tile(k, [1, repeats, 1, 1])
            v = mx.tile(v, [1, repeats, 1, 1])

        self._cached_k = k
        self._cached_v = v

    def clear_kv(self) -> None:
        self._cached_k = None
        self._cached_v = None

    def __call__(
        self,
        x: mx.array,
        encoder_out: mx.array | None = None,
    ) -> mx.array:
        batch_size, query_len, _ = x.shape

        q = self.to_q(x).reshape(
            batch_size, query_len, self.heads, self.head_dim
        )
        q = q.transpose(0, 2, 1, 3)
        q = _l2_normalize(q) * self.qk_norm_q_scale

        if self._cached_k is not None:
            k = self._cached_k
            v = self._cached_v
        else:
            assert encoder_out is not None
            self.precompute_kv(encoder_out)
            k = self._cached_k
            v = self._cached_v

        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.qk_norm_scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        out = mx.matmul(weights.astype(v.dtype), v)
        out = out.transpose(0, 2, 1, 3).reshape(
            batch_size, query_len, self.heads * self.head_dim
        )
        return self.to_out(out)


class MLXSwiGLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2)
        self.out = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        value, gate = mx.split(self.proj(x), 2, axis=-1)
        return self.out(value * nn.silu(gate))


class MLXTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        ff_mult: int = 4,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        self.attn_norm = MLXSimpleRMSNorm(dim)
        self.attn = MLXSelfAttention(
            dim=dim,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            qk_norm_scale=qk_norm_scale,
        )
        self.ff_norm = MLXSimpleRMSNorm(dim)
        self.ff = MLXSwiGLU(dim, dim * ff_mult)

    def __call__(
        self,
        x: mx.array,
        rotary_freqs: mx.array | None,
        rel_pos: MLXRelativePositionBias | None = None,
        cache: MLXKVCache | None = None,
        layer_idx: int = 0,
        causal: bool = True,
    ) -> mx.array:
        x = x + self.attn(
            self.attn_norm(x),
            rotary_freqs=rotary_freqs,
            rel_pos=rel_pos,
            cache=cache,
            layer_idx=layer_idx,
            causal=causal,
        )
        x = x + self.ff(self.ff_norm(x))
        return x


class MLXEncoderDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        ff_mult: int = 4,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        self.self_attn_norm = MLXSimpleRMSNorm(dim)
        self.self_attn = MLXSelfAttention(
            dim=dim,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            qk_norm_scale=qk_norm_scale,
        )
        self.cross_attn_norm = MLXSimpleRMSNorm(dim)
        self.cross_attn = MLXCrossAttention(
            dim=dim,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            qk_norm_scale=qk_norm_scale,
        )
        self.ff_norm = MLXSimpleRMSNorm(dim)
        self.ff = MLXSwiGLU(dim, dim * ff_mult)

    def __call__(
        self,
        x: mx.array,
        rotary_freqs: mx.array | None,
        rel_pos: MLXRelativePositionBias | None = None,
        encoder_out: mx.array | None = None,
        cache: MLXKVCache | None = None,
        layer_idx: int = 0,
        causal: bool = True,
    ) -> mx.array:
        x = x + self.self_attn(
            self.self_attn_norm(x),
            rotary_freqs=rotary_freqs,
            rel_pos=rel_pos,
            cache=cache,
            layer_idx=layer_idx,
            causal=causal,
        )
        x = x + self.cross_attn(self.cross_attn_norm(x), encoder_out)
        x = x + self.ff(self.ff_norm(x))
        return x


class MLXDecoderCore(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        kv_heads: int,
        num_tokens: int,
        head_dim: int | None = None,
        ff_mult: int = 4,
        rotary_pos_emb: bool = True,
        rotary_emb_dim: int | None = None,
        rel_pos_bias: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        head_dim = head_dim or dim // heads
        rotary_emb_dim = rotary_emb_dim or head_dim // 2

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim

        self.blocks = [
            MLXTransformerBlock(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                qk_norm_scale=qk_norm_scale,
            )
            for _ in range(depth)
        ]
        self.final_norm = MLXSimpleRMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.rotary_pos_emb = (
            MLXRotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )
        self.rel_pos = (
            MLXRelativePositionBias(
                scale=math.sqrt(head_dim),
                causal=True,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
            if rel_pos_bias
            else None
        )

    def __call__(
        self,
        x: mx.array,
        cache: MLXKVCache | None = None,
    ) -> mx.array:
        seq_len = x.shape[1]
        rotary_freqs = None

        if self.rotary_pos_emb is not None:
            offset = 0 if cache is None else cache.offset
            positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
            rotary_freqs = self.rotary_pos_emb(positions)

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                rotary_freqs=rotary_freqs,
                rel_pos=self.rel_pos,
                cache=cache,
                layer_idx=layer_idx,
                causal=True,
            )

        x = self.final_norm(x)

        if cache is not None:
            cache.advance(seq_len)

        return self.to_logits(x)

    def make_cache(self) -> MLXKVCache:
        return MLXKVCache(self.depth)


class MLXEncoderCore(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        kv_heads: int,
        head_dim: int | None = None,
        ff_mult: int = 4,
        rotary_pos_emb: bool = False,
        rotary_emb_dim: int | None = None,
        rel_pos_bias: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        head_dim = head_dim or dim // heads
        rotary_emb_dim = rotary_emb_dim or head_dim // 2

        self.blocks = [
            MLXTransformerBlock(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                qk_norm_scale=qk_norm_scale,
            )
            for _ in range(depth)
        ]
        self.final_norm = MLXSimpleRMSNorm(dim)
        self.rotary_pos_emb = (
            MLXRotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )
        self.rel_pos = (
            MLXRelativePositionBias(
                scale=math.sqrt(head_dim),
                causal=False,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
            if rel_pos_bias
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        rotary_freqs = None
        if self.rotary_pos_emb is not None:
            positions = mx.arange(x.shape[1], dtype=mx.float32)
            rotary_freqs = self.rotary_pos_emb(positions)

        for block in self.blocks:
            x = block(
                x,
                rotary_freqs=rotary_freqs,
                rel_pos=self.rel_pos,
                cache=None,
                layer_idx=0,
                causal=False,
            )
        return self.final_norm(x)


class MLXCrossAttnDecoderCore(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        kv_heads: int,
        num_tokens: int,
        head_dim: int | None = None,
        ff_mult: int = 4,
        rotary_pos_emb: bool = False,
        rotary_emb_dim: int | None = None,
        rel_pos_bias: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
        qk_norm_scale: float = 10.0,
    ):
        super().__init__()
        head_dim = head_dim or dim // heads
        rotary_emb_dim = rotary_emb_dim or head_dim // 2

        self.depth = depth
        self.blocks = [
            MLXEncoderDecoderBlock(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
                qk_norm_scale=qk_norm_scale,
            )
            for _ in range(depth)
        ]
        self.final_norm = MLXSimpleRMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        self.rotary_pos_emb = (
            MLXRotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )
        self.rel_pos = (
            MLXRelativePositionBias(
                scale=math.sqrt(head_dim),
                causal=True,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
            if rel_pos_bias
            else None
        )

    def precompute_cross_attn_kv(self, encoder_out: mx.array) -> None:
        for block in self.blocks:
            block.cross_attn.precompute_kv(encoder_out)

    def clear_cross_attn_kv(self) -> None:
        for block in self.blocks:
            block.cross_attn.clear_kv()

    def __call__(
        self,
        x: mx.array,
        encoder_out: mx.array | None = None,
        cache: MLXKVCache | None = None,
    ) -> mx.array:
        seq_len = x.shape[1]
        rotary_freqs = None
        if self.rotary_pos_emb is not None:
            offset = 0 if cache is None else cache.offset
            positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
            rotary_freqs = self.rotary_pos_emb(positions)

        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                rotary_freqs=rotary_freqs,
                rel_pos=self.rel_pos,
                encoder_out=encoder_out,
                cache=cache,
                layer_idx=layer_idx,
                causal=True,
            )

        x = self.final_norm(x)

        if cache is not None:
            cache.advance(seq_len)

        return self.to_logits(x)

    def make_cache(self) -> MLXKVCache:
        return MLXKVCache(self.depth)


def _map_ff_param(prefix: str, param_path: str) -> str | None:
    if param_path.startswith("ff.0.proj."):
        suffix = param_path[len("ff.0.proj.") :]
        return f"{prefix}.proj.{suffix}"
    if param_path.startswith("ff.2."):
        suffix = param_path[len("ff.2.") :]
        return f"{prefix}.out.{suffix}"
    if param_path.startswith("ff.0.0."):
        suffix = param_path[len("ff.0.0.") :]
        return f"{prefix}.proj.{suffix}"
    return None


def _map_decoder_key(key: str) -> str | None:
    if key == "attn_layers.rotary_pos_emb.inv_freq":
        return None
    if key == "attn_layers.rel_pos.relative_attention_bias.weight":
        return "rel_pos.relative_attention_bias.weight"
    if key == "to_logits.weight":
        return "to_logits.weight"
    if key == "to_logits.bias":
        return "to_logits.bias"

    if not key.startswith("attn_layers.layers."):
        return None

    rest = key[len("attn_layers.layers.") :]
    parts = rest.split(".", 2)
    if len(parts) < 3 or parts[1] != "1":
        return None

    flat_idx = int(parts[0])
    param_path = parts[2]
    block_idx = flat_idx // 2
    if flat_idx % 2 == 0:
        return f"blocks.{block_idx}.attn.{param_path}"
    return _map_ff_param(f"blocks.{block_idx}.ff", param_path)


def _map_cross_attn_decoder_key(key: str) -> str | None:
    if key == "attn_layers.rotary_pos_emb.inv_freq":
        return None
    if key == "attn_layers.rel_pos.relative_attention_bias.weight":
        return "rel_pos.relative_attention_bias.weight"
    if key == "to_logits.weight":
        return "to_logits.weight"
    if key == "to_logits.bias":
        return "to_logits.bias"

    if not key.startswith("attn_layers.layers."):
        return None

    rest = key[len("attn_layers.layers.") :]
    parts = rest.split(".", 2)
    if len(parts) < 3 or parts[1] != "1":
        return None

    flat_idx = int(parts[0])
    param_path = parts[2]
    block_idx = flat_idx // 3
    sub_idx = flat_idx % 3

    if sub_idx == 0:
        return f"blocks.{block_idx}.self_attn.{param_path}"
    if sub_idx == 1:
        return f"blocks.{block_idx}.cross_attn.{param_path}"
    return _map_ff_param(f"blocks.{block_idx}.ff", param_path)


def _map_encoder_key(key: str) -> str | None:
    if key == "attn_layers.rotary_pos_emb.inv_freq":
        return None
    if key == "attn_layers.rel_pos.relative_attention_bias.weight":
        return "rel_pos.relative_attention_bias.weight"

    if not key.startswith("attn_layers.layers."):
        return None

    rest = key[len("attn_layers.layers.") :]
    parts = rest.split(".", 2)
    if len(parts) < 3 or parts[1] != "1":
        return None

    flat_idx = int(parts[0])
    param_path = parts[2]
    block_idx = flat_idx // 2
    if flat_idx % 2 == 0:
        return f"blocks.{block_idx}.attn.{param_path}"
    return _map_ff_param(f"blocks.{block_idx}.ff", param_path)


def convert_decoder_weights(
    state_dict: dict,
    to_mx: bool = True,
) -> dict[str, mx.array]:
    converted = {}
    for key, value in state_dict.items():
        new_key = _map_decoder_key(key)
        if new_key is not None:
            converted[new_key] = _torch_to_mx(value) if to_mx else value
    return converted


def convert_cross_attn_decoder_weights(
    state_dict: dict,
    to_mx: bool = True,
) -> dict[str, mx.array]:
    converted = {}
    for key, value in state_dict.items():
        new_key = _map_cross_attn_decoder_key(key)
        if new_key is not None:
            converted[new_key] = _torch_to_mx(value) if to_mx else value
    return converted


def convert_encoder_weights(
    state_dict: dict,
    to_mx: bool = True,
) -> dict[str, mx.array]:
    converted = {}
    for key, value in state_dict.items():
        new_key = _map_encoder_key(key)
        if new_key is not None:
            converted[new_key] = _torch_to_mx(value) if to_mx else value
    return converted


def load_mlx_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    strict: bool = False,
) -> None:
    model.load_weights(list(weights.items()), strict=strict)
