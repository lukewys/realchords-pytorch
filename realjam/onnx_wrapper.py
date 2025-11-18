import torch
import torch.nn as nn
from x_transformers.x_transformers import LayerIntermediates, Intermediates
import numpy as np


class EncoderWrapper(nn.Module):
    """Encoder forward with *only* enc_inputs (B,60) → encodings (B,60,D)."""

    def __init__(self, enc_model):
        super().__init__()
        self.enc = enc_model

    def forward(self, enc_packed):
        enc_inputs = enc_packed[..., 0]
        enc_mask = enc_packed[..., 1].bool()
        return self.enc(enc_inputs, mask=enc_mask, return_embeddings=True)

    __call__ = forward


class ModelExportWrapper(nn.Module):
    """
    Wraps your model so we have a single forward(...) signature
    that matches the inputs and outputs we want for ONNX export.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        gen_inputs,
        hiddens,
        last_hidden,
        values,
        cached_kv0,
        cached_kv1,
        layer_hiddens,
        attn_z_loss,
        mems,
        memory_tokens,
        logit_entropies,
    ):
        hiddens = torch.split(hiddens, 1)
        hiddens = [torch.squeeze(x, 0) for x in hiddens]
        mask = gen_inputs[..., -1].bool()
        gen_inputs = gen_inputs[..., 0]

        layer_hiddens = torch.split(layer_hiddens, 1)
        layer_hiddens = [torch.squeeze(x, 0) for x in layer_hiddens]

        values = torch.split(values, 1)
        values = [torch.squeeze(x, 0) for x in values]

        cached_kv0 = torch.split(cached_kv0, 1)
        cached_kv0 = [torch.squeeze(x, 0) for x in cached_kv0]

        cached_kv1 = torch.split(cached_kv1, 1)
        cached_kv1 = [torch.squeeze(x, 0) for x in cached_kv1]

        cached_kv = [
            (x[:, :, 1:, :], y[:, :, 1:, :])
            for x, y in zip(cached_kv0, cached_kv1)
        ]

        attn_intermediates = [
            Intermediates(values=x, cached_kv=y)
            for x, y in zip(values, cached_kv)
        ]

        # Reconstruct the "cache" object
        cache = LayerIntermediates(
            hiddens=hiddens,
            last_hidden=last_hidden,
            attn_intermediates=attn_intermediates,
            layer_hiddens=layer_hiddens,
        )
        # Call the underlying model
        # Depending on your model’s signature, adapt the below line
        out, cache_out = self.base_model(
            gen_inputs,
            cache=cache,
            return_intermediates=True,
            self_attn_kv_mask=mask,
        )

        cache_out_dict = cache_out.__dict__
        hiddens = cache_out_dict["hiddens"]
        hiddens = torch.stack(hiddens)
        last_hidden = cache_out_dict["last_hidden"]
        attn_intermediates = cache_out_dict["attn_intermediates"]
        values = [x.values for x in attn_intermediates]
        cached_kv0 = [x.cached_kv[0] for x in attn_intermediates]
        cached_kv1 = [x.cached_kv[1] for x in attn_intermediates]

        values = torch.stack(values)
        cached_kv0 = torch.stack(cached_kv0)
        cached_kv1 = torch.stack(cached_kv1)

        layer_hiddens = cache_out_dict["layer_hiddens"]
        layer_hiddens = torch.stack(layer_hiddens)
        return [
            out,
            hiddens,
            last_hidden,
            values,
            cached_kv0,
            cached_kv1,
            layer_hiddens,
        ]

    def __call__(
        self,
        gen_inputs,
        hiddens,
        last_hidden,
        values,
        cached_kv0,
        cached_kv1,
        layer_hiddens,
        attn_z_loss=None,
        mems=None,
        memory_tokens=None,
        logit_entropies=None,
    ):
        return self.forward(
            gen_inputs,
            hiddens,
            last_hidden,
            values,
            cached_kv0,
            cached_kv1,
            layer_hiddens,
            attn_z_loss,
            mems,
            memory_tokens,
            logit_entropies,
        )


class OnlineWrapperStep(nn.Module):
    """
    Wraps your model so we have a single forward(...) signature
    that matches the inputs and outputs we want for ONNX export.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        packed_inputs,
        hiddens,
        last_hidden,
        values,
        cached_kv0,
        cached_kv1,
        layer_hiddens,
        attn_z_loss,
        mems,
        memory_tokens,
        logit_entropies,
    ):
        hiddens = torch.split(hiddens, 1)
        hiddens = [torch.squeeze(x, 0) for x in hiddens]

        mask = packed_inputs[..., -1].bool()
        gen_inputs = packed_inputs[..., 0]

        layer_hiddens = torch.split(layer_hiddens, 1)
        layer_hiddens = [torch.squeeze(x, 0) for x in layer_hiddens]

        values = torch.split(values, 1)
        values = [torch.squeeze(x, 0) for x in values]

        cached_kv0 = torch.split(cached_kv0, 1)
        cached_kv0 = [torch.squeeze(x, 0) for x in cached_kv0]

        cached_kv1 = torch.split(cached_kv1, 1)
        cached_kv1 = [torch.squeeze(x, 0) for x in cached_kv1]

        cached_kv = []
        for x, y in zip(cached_kv0, cached_kv1):
            x = x[:, :, 1:, :]
            y = y[:, :, 1:, :]
            cached_kv.append((x, y))

        attn_intermediates = [
            Intermediates(values=x, cached_kv=y)
            for x, y in zip(values, cached_kv)
        ]

        # Reconstruct the "cache" object
        cache = LayerIntermediates(
            hiddens=hiddens,
            last_hidden=last_hidden,
            attn_intermediates=attn_intermediates,
            layer_hiddens=layer_hiddens,
            # attn_z_loss=attn_z_loss,
            # mems=mems,
            # memory_tokens=memory_tokens,
            # logit_entropies=logit_entropies,
        )
        # Call the underlying model
        # Depending on your model’s signature, adapt the below line
        out, cache_out = self.base_model(
            gen_inputs,
            cache=cache,
            return_intermediates=True,
            self_attn_kv_mask=mask,
        )

        cache_out_dict = cache_out.__dict__
        hiddens = cache_out_dict["hiddens"]
        hiddens = torch.stack(hiddens)
        last_hidden = cache_out_dict["last_hidden"]
        attn_intermediates = cache_out_dict["attn_intermediates"]
        values = [x.values for x in attn_intermediates]

        values = torch.stack(values)
        cached_kv0 = torch.stack(cached_kv0)
        cached_kv1 = torch.stack(cached_kv1)

        layer_hiddens = cache_out_dict["layer_hiddens"]
        layer_hiddens = torch.stack(layer_hiddens)
        return [
            out,
            hiddens,
            last_hidden,
            values,
            cached_kv0,
            cached_kv1,
            layer_hiddens,
        ]

    def __call__(
        self,
        gen_inputs,
        hiddens,
        last_hidden,
        values,
        cached_kv0,
        cached_kv1,
        layer_hiddens,
        attn_z_loss=None,
        mems=None,
        memory_tokens=None,
        logit_entropies=None,
    ):
        return self.forward(
            gen_inputs,
            hiddens,
            last_hidden,
            values,
            cached_kv0,
            cached_kv1,
            layer_hiddens,
            attn_z_loss,
            mems,
            memory_tokens,
            logit_entropies,
        )


class OnlineWrapperInit(nn.Module):
    """
    Wraps your model so we have a single forward(...) signature
    that matches the inputs and outputs we want for ONNX export.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        packed_inputs,
    ):

        enc = None
        enc_mask = None
        if packed_inputs.shape[-1] == 2:
            mask = packed_inputs[..., -1].bool()
            gen_inputs = packed_inputs[..., 0]
        else:
            mask = packed_inputs[..., 1].bool()
            gen_inputs = packed_inputs[..., 0].long()
            enc_mask = packed_inputs[..., 2].bool()
            enc = packed_inputs[..., 3:].float()
        out, cache_out = self.base_model(
            gen_inputs,
            cache=None,
            mask=mask,
            return_intermediates=True,
            context=enc,
            context_mask=enc_mask,
        )

        cache_out_dict = cache_out.__dict__
        hiddens = cache_out_dict["hiddens"]
        hiddens = torch.stack(hiddens)
        last_hidden = cache_out_dict["last_hidden"]
        attn_intermediates = cache_out_dict["attn_intermediates"]
        values = [x.values for x in attn_intermediates]
        cached_kv0 = [x.cached_kv[0] for x in attn_intermediates]
        cached_kv1 = [x.cached_kv[1] for x in attn_intermediates]

        values = torch.stack(values)
        cached_kv0 = torch.stack(cached_kv0)
        cached_kv1 = torch.stack(cached_kv1)

        layer_hiddens = cache_out_dict["layer_hiddens"]
        layer_hiddens = torch.stack(layer_hiddens)
        return [
            out,
            hiddens,
            last_hidden,
            values,
            cached_kv0,
            cached_kv1,
            layer_hiddens,
        ]

    def __call__(self, gen_inputs):

        return self.forward(gen_inputs)
