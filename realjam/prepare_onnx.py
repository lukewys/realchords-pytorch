# ReaLchords/utils/prepare_online.py
# --------------------------------------------------------------------
"""
Export the *online* causal decoder (init + step graphs) to ONNX

Usage
-----
>>> from realchords.utils.prepare_online import prepare_online
>>> sess_init, sess_step = prepare_online(
...         model, tokenizer,
...         max_gen_seq_len=512,
...         save_dir="checkpoints/onnx")
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort

from realjam.onnx_wrapper import (
    OnlineWrapperInit,
    OnlineWrapperStep,
    EncoderWrapper,
    ModelExportWrapper,
)


# ------------------------------------------------------------------ #
# public convenience                                                 #
# ------------------------------------------------------------------ #
def prepare_online(
    model: torch.nn.Module,
    tokenizer,  # only for pad_token lookup
    *,
    max_gen_seq_len: int = 512,
    save_dir: str | Path = "checkpoints/onnx",
    opset: int = 14,
    model_name: str = "online",
    overwrite: bool = False,
    provider: str = "CPUExecutionProvider",
) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
    """
    Export *init* and *step* ONNX graphs if they don’t exist yet, then
    return the loaded `onnxruntime.InferenceSession`s.

    Parameters
    ----------
    model : torch.nn.Module
        The trained `LitDecoder` already loaded with weights.
    tokenizer
        Only used for its `.pad_token` ID when padding dummy input.
    max_gen_seq_len : int
        The fixed 1-window length baked into the ONNX graph (512 by default).
    save_dir : str | Path
        Directory where `model_online_kv_{init,step}.onnx` will live.
    opset : int
        ONNX opset version.

    Returns
    -------
    (sess_init, sess_step)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fp_init = save_dir / f"model_{model_name}_{max_gen_seq_len}_kv_init.onnx"
    fp_step = save_dir / f"model_{model_name}_{max_gen_seq_len}_kv.onnx"

    if not fp_init.exists() or overwrite:
        _export_online_init(model, fp_init, max_gen_seq_len, opset)

    if not fp_step.exists() or overwrite:
        _export_online_step(
            model,
            tokenizer,
            fp_step,
            max_gen_seq_len,
            opset,
        )

    sess_init = ort.InferenceSession(fp_init.as_posix(), providers=[provider])
    sess_step = ort.InferenceSession(fp_step.as_posix(), providers=[provider])
    return sess_init, sess_step


# ------------------------------------------------------------------ #
# internals                                                          #
# ------------------------------------------------------------------ #
def _export_online_init(
    model: torch.nn.Module,
    file_path: Path,
    max_len: int,
    opset: int,
    context: bool = False,
) -> None:
    """
    Export the *first-frame* graph (no KV cache fed in).
    """
    print(f"[ONNX]  exporting init graph → {file_path}")
    if context:
        dummy = torch.zeros((1, max_len, 3 + 512), dtype=torch.long)
        dummy[:, :, 2] = 1
    else:
        dummy = torch.zeros((1, max_len, 2), dtype=torch.long)

    wrapper = OnlineWrapperInit(model).eval()
    torch.onnx.export(
        wrapper,
        (dummy,),
        file_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        input_names=["gen_inputs"],
        output_names=[
            "logits",
            "hiddens",
            "last_hidden",
            "values",
            "cached_kv0",
            "cached_kv1",
            "layer_hiddens",
        ],
        dynamic_axes={"gen_inputs": {0: "batch_size"}},
        dynamo=False,
    )


def _export_online_step(
    model: torch.nn.Module,
    tokenizer,
    file_path: Path,
    max_len: int,
    opset: int,
    context: bool = False,
) -> None:
    """
    Export the *subsequent-frame* graph that consumes / produces KV cache.
    """
    print(f"[ONNX]  exporting step graph → {file_path}")

    # ---------------- build a **real** cache by running one dummy pass ----
    gen_inputs = torch.zeros((1, 4), dtype=torch.long)  # 4-token toy prompt
    pad_needed = max_len - gen_inputs.shape[1]
    if pad_needed > 0:
        gen_inputs = F.pad(
            gen_inputs,
            (0, pad_needed),
            value=tokenizer.pad_token,
        )

    mask = torch.zeros_like(gen_inputs, dtype=torch.bool)

    # run once to get realistic cache tensors
    if context:
        dummy_enc = torch.ones((1, max_len, 512), dtype=torch.float)
        enc_mask = torch.ones_like(mask, dtype=torch.bool)
        _, cache = model.forward(
            gen_inputs,
            mask=mask,
            context=dummy_enc,
            context_mask=enc_mask,
            return_intermediates=True,
        )
    else:
        _, cache = model.forward(
            gen_inputs, mask=mask, return_intermediates=True
        )
    cd = cache.__dict__

    # ---------------- collect tensors in wrapper-expected order ------------
    hiddens = torch.stack(cd["hiddens"])
    last_hidden = cd["last_hidden"]
    layer_h = torch.stack(cd["layer_hiddens"])
    vals = torch.stack([x.values for x in cd["attn_intermediates"]])
    kv0 = torch.stack([x.cached_kv[0] for x in cd["attn_intermediates"]])
    kv1 = torch.stack([x.cached_kv[1] for x in cd["attn_intermediates"]])
    layer_type = [x.layer_type for x in cd["attn_intermediates"]]

    if not context:
        inp_pack = torch.stack([gen_inputs, mask], dim=-1)
    else:
        inp_pack = torch.stack([gen_inputs, mask, enc_mask], dim=-1)
        inp_pack = torch.cat([inp_pack, dummy_enc], dim=-1)

    # ---------------- export ------------------------------------------------
    wrapper = ModelExportWrapper(model).eval()
    torch.onnx.export(
        wrapper,
        (
            inp_pack,
            hiddens,
            last_hidden,
            vals,
            kv0,
            kv1,
            layer_h,
        ),
        file_path.as_posix(),
        export_params=True,
        opset_version=opset,
        do_constant_folding=False,
        input_names=[
            "gen_inputs",
            "hiddens",
            "last_hidden",
            "values",
            "cached_kv0",
            "cached_kv1",
            "layer_hiddens",
        ],
        output_names=["decoder_preds"],
        dynamic_axes={
            "gen_inputs": {0: "batch_size"},
            "hiddens": {1: "batch_size"},
            "last_hidden": {0: "batch_size"},
            "values": {1: "batch_size"},
            "cached_kv0": {1: "batch_size"},
            "cached_kv1": {1: "batch_size"},
            "layer_hiddens": {1: "batch_size"},
            "decoder_preds": {0: "batch_size"},
        },
        dynamo=False,
    )


def _export_encoder(model, fp, opset=14, enc_length=60):
    dummy = torch.zeros((1, enc_length), dtype=torch.long)
    mask = torch.zeros_like(dummy, dtype=torch.bool)
    enc_pack = torch.stack([dummy, mask], dim=-1)  # (1, ENC_LEN, 2)
    torch.onnx.export(
        EncoderWrapper(model.encoder).eval(),
        enc_pack,
        fp.as_posix(),
        input_names=["enc_inputs", "enc_mask"],
        output_names=["encodings"],
        dynamic_axes={"enc_inputs": {0: "batch"}, "enc_mask": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=False,
        dynamo=False,
    )


def prepare_offline60(
    model,
    tokenizer,
    save_dir="checkpoints/onnx60",
    overwrite=False,
    provider="CPUExecutionProvider",
    opset=14,
):
    sd = Path(save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    fenc = sd / "encoder60.onnx"
    finit = sd / "decoder60_init.onnx"
    fstep = sd / "decoder60_step.onnx"

    if overwrite or not fenc.exists():
        _export_encoder(model, fenc)
    if overwrite or not finit.exists():
        _export_online_init(
            model.decoder, finit, context=True, max_len=ENC_LEN, opset=opset
        )
    if overwrite or not fstep.exists():
        _export_online_step(
            model.decoder,
            tokenizer,
            fstep,
            context=True,
            max_len=ENC_LEN,
            opset=opset,
        )

    ep = [provider]
    return (
        ort.InferenceSession(fenc.as_posix(), providers=ep),
        ort.InferenceSession(finit.as_posix(), providers=ep),
        ort.InferenceSession(fstep.as_posix(), providers=ep),
    )
