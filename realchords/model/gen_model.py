"""Generative models for ReaLchords."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from typing import Tuple, Callable, List
from einops import rearrange, pack, unpack
from tqdm import tqdm

from x_transformers import TransformerWrapper, AutoregressiveWrapper
from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    exists,
    join,
    align_right,
)

from realchords.nn.transformers import Decoder, Encoder
from realchords.model.sampling import (
    top_k,
    FILTER_LOGITS_FN,
    ComposeFilterFns,
    validate_filter_fn_kwargs,
)


class DecoderTransformer(AutoregressiveWrapper):
    """Decoder-only transformer model.

    We add the generation function for online/synchronous generation under
    the case of model-data (single-agent) and model-model (multi-agent).
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_value: int = 0,
        cross_attend: bool = False,
        attention_layer_configs: dict = dict(),
        transformer_configs: dict = dict(),
    ):
        # Skip the init of the AutoregressiveWrapper class
        super(AutoregressiveWrapper, self).__init__()
        self.decoder = TransformerWrapper(
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
                cross_attend=cross_attend,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            **transformer_configs,
        )

        # Save arguments
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_value = pad_value

    @property
    def net(self):
        return self.decoder

    def forward(self, x, mask=None, **kwargs):
        return self.decoder(x, mask=mask, **kwargs)

    @torch.no_grad()
    @torch.jit.export
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable | List[str | Callable] = top_k,
        restrict_to_max_seq_len=True,
        filter_kwargs: dict | List[dict] = dict(),
        cache_kv=True,
        display_pbar=False,
        **kwargs,
    ):
        """Generate Function modified from AutoregressiveWrapper.

        Add the following supports:
        1. Input current sampling step to filter_logits_fn.
        2. support masked sampling.
        3. Support tqdm progress bar.
        4. Input history sampling step to filter_logits_fn.

        Remove the following supports:
        1. contrastive decoding
        """
        # handle multiple filter logits functions
        filter_fns_is_list = validate_filter_fn_kwargs(
            filter_logits_fn, filter_kwargs
        )
        if filter_fns_is_list:
            filter_fns = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_logits_fn = filter_fns
            filter_kwargs = dict()

        max_seq_len, greedy, device = (
            self.max_seq_len,
            temperature == 0.0,
            prompts.device,
        )

        b, t = prompts.shape

        # handle filter logits fn given as string
        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)
        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens are appended
        out = prompts

        # kv cache
        cache = None

        pbar = tqdm(
            range(seq_len),
            disable=not display_pbar,
            desc="Sampling",
        )

        for curr_sample_step in pbar:
            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len
                assert not (
                    cache_kv
                    and max_len_exceeded
                    and not self.net.can_cache_kv_outside_max_seq_len
                ), (
                    "the network cannot use cached key values when decoding outside "
                    "the max sequence length. consider switching to rotary embeddings."
                )
                x = out[:, -max_seq_len:]
                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == "a":
                            inter.cached_kv = [
                                t[..., -(max_seq_len - 1) :, :]
                                for t in inter.cached_kv
                            ]
            else:
                x = out

            logits, new_cache = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs,
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]  # get logits for the last token

            # Sampling step
            if greedy:
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                curr_sample_length = out.shape[-1]
                filtered_logits = filter_logits_fn(
                    logits,
                    curr_sample_step=curr_sample_step,
                    curr_sample_length=curr_sample_length,
                    curr_sequence=out,
                    **filter_kwargs,
                )
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(dim=-1).all():
                    break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # Remove the prompt tokens
        out = out[:, t:]

        return out

    @torch.no_grad()
    @eval_decorator
    def generate_online(
        self,
        prompts,
        conditions,  # condition for model (the other part)
        seq_len,
        eos_token=None,
        temperature=1.0,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable | List[str | Callable] = top_k,
        restrict_to_max_seq_len=True,
        filter_kwargs: dict | List[dict] = dict(),
        cache_kv=True,
        reverse_condition_order=False,
        **kwargs,
    ):
        """Generate tokens in online ReaLchords manner.

        For online generation, we will first generate a token from model,
            then fill the next token from conditions, forming a generation loop
            that "simulates" the online generation with a fixed data.

        Note the condition input will be the condition filled for each generation step,
            but not the actual resulting sequence step.
        Thus, if there are non empty prompts, the condition input will be for
            the generation steps after the prompts.

        Removed features compared to x_transformers.AutoregressiveWrapper.generate:
        1. contrastive decoding

        if reverse_condition_order is True, the condition will be placed before the prompt.
        So that the model will be predicting p(yt | y<t, x<=t) instead of p(yt | y<t, x<t).
        """

        # handle multiple filter logits functions
        filter_fns_is_list = validate_filter_fn_kwargs(
            filter_logits_fn, filter_kwargs
        )
        if filter_fns_is_list:
            filter_fns = ComposeFilterFns(filter_logits_fn, filter_kwargs)
            filter_logits_fn = filter_fns
            filter_kwargs = dict()

        max_seq_len, greedy, device = (
            self.max_seq_len,
            temperature == 0.0,
            prompts.device,
        )

        prompts, ps = pack([prompts], "* n")

        b, t = prompts.shape

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert (
                filter_logits_fn in FILTER_LOGITS_FN
            ), f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        if reverse_condition_order:
            check_is_model_part = lambda x: x % 2 == 1
        else:
            check_is_model_part = lambda x: x % 2 == 0

        # sampling up to seq_len

        for step in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (
                    cache_kv
                    and max_len_exceeded
                    and not self.net.can_cache_kv_outside_max_seq_len
                ), (
                    "the network cannot use cached key values when decoding outside the max sequence length. "
                    "most likely because you are using absolute positional embedding. "
                    "you can switch to rotary embeddings to resolve this issue"
                )

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == "a":
                            inter.cached_kv = [
                                t[..., -(max_seq_len - 1) :, :]
                                for t in inter.cached_kv
                            ]

            if check_is_model_part(step):
                # Sample from model

                logits, new_cache = self.net(
                    x,
                    return_intermediates=True,
                    cache=cache,
                    seq_start_pos=seq_start_pos,
                    **kwargs,
                )

                if cache_kv and self.net.can_cache_kv:
                    cache = new_cache

                logits = logits[:, -1]

                # filter by top_k, top_p (nucleus), top_a, or custom

                if greedy:
                    sample = logits.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits = filter_logits_fn(
                        logits,
                        curr_sample_step=step,
                        curr_sample_length=out.shape[-1],
                        curr_sequence=out,
                        **filter_kwargs,
                    )
                    probs = F.softmax(filtered_logits / temperature, dim=-1)
                    sample = torch.multinomial(probs, 1)

            else:
                # Concat the condition to the sample
                sample = conditions[:, step // 2].unsqueeze(-1)
                if cache_kv and self.net.can_cache_kv:
                    # go though net just to refresh the cache
                    _, new_cache = self.net(
                        x,
                        return_intermediates=True,
                        cache=cache,
                        seq_start_pos=seq_start_pos,
                        **kwargs,
                    )
                    cache = new_cache

            # concat sample

            out = torch.cat((out, sample), dim=-1)

            if not exists(eos_token):
                continue

            is_eos_tokens = out == eos_token

            if is_eos_tokens.any(dim=-1).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        out = out[:, t:]

        (out,) = unpack(out, ps, "* n")

        return out


class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder transformer model.

    This class is almost the same as x_transformers.XTransformer, but leaves
    flexibility to modify the network architecture and condition input method.
    """

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
        dropout: float = 0.1,
        pad_value: int = 0,
        attention_layer_configs: dict = dict(),
        transformer_configs: dict = dict(),
    ):
        super().__init__()
        self.encoder = TransformerWrapper(
            attn_layers=Encoder(
                dim=enc_dim,
                depth=enc_depth,
                heads=enc_heads,
                dropout=dropout,
                attention_layer_configs=attention_layer_configs,
            ),
            num_tokens=enc_num_tokens,
            max_seq_len=enc_max_seq_len,
            return_only_embed=True,
            **transformer_configs,
        )
        self.decoder = DecoderTransformer(
            dim=dec_dim,
            depth=dec_depth,
            heads=dec_heads,
            num_tokens=dec_num_tokens,
            max_seq_len=dec_max_seq_len,
            dropout=dropout,
            pad_value=pad_value,
            cross_attend=True,
            attention_layer_configs=attention_layer_configs,
            transformer_configs=transformer_configs,
        )

        # Save arguments
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
        self.dropout = dropout
        self.pad_value = pad_value

    def forward(
        self,
        x_enc,
        x_dec,
        enc_mask=None,
        dec_mask=None,
        return_attn_z_loss=False,
    ):
        if return_attn_z_loss:
            enc, cache = self.encoder(
                x_enc,
                mask=enc_mask,
                return_embeddings=True,
                return_attn_z_loss=True,
            )
            z_loss_enc = cache.attn_z_loss
            dec, cache = self.decoder(
                x_dec,
                context=enc,
                context_mask=enc_mask,
                mask=dec_mask,
                return_attn_z_loss=True,
            )
            z_loss_dec = cache.attn_z_loss
            return dec, z_loss_enc + z_loss_dec
        else:
            enc = self.encoder(x_enc, mask=enc_mask, return_embeddings=True)
            dec = self.decoder(
                x_dec, context=enc, context_mask=enc_mask, mask=dec_mask
            )
            return dec

    @torch.no_grad()
    def generate(
        self,
        seq_in,
        seq_out_start,
        seq_len,
        mask=None,
        attn_mask=None,
        **kwargs,
    ):
        encodings = self.encoder(
            seq_in, mask=mask, attn_mask=attn_mask, return_embeddings=True
        )
        return self.decoder.generate(
            seq_out_start,
            seq_len,
            context=encodings,
            context_mask=mask,
            **kwargs,
        )
