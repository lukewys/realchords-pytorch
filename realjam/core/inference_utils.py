"""Utilities for loading models for inference."""

import copy
import torch
from typing import Optional
from pathlib import Path
import argbind


def load_rl_model(
    model_path: str,
    model: torch.nn.Module,
    compile: bool = True,
):
    """Load RL-finetuned model from a given model path."""
    # Note that the model input should not be compiled.
    # check if the model is compiled
    if list(model.state_dict().keys())[0].startswith("_orig_mod"):
        raise ValueError("The model should not be compiled.")

    model_path = Path(model_path)
    state_dict = torch.load(
        model_path, weights_only=True, map_location=torch.device("cpu")
    )
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    state_dict_model = {}
    for k, v in state_dict.items():
        if k.startswith("model.module."):
            k = k.replace("model.module.", "")
            state_dict_model[k] = v
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(state_dict_model)
    if compile:
        new_model = torch.compile(new_model)
    return new_model


def load_gen_model_from_state_dict(
    state_dict_path: str,
    model_cls: torch.nn.Module,
    compile: bool = True,
    override_args: Optional[dict] = None,
):
    """Load generative model from a given state dict path."""
    state_dict = torch.load(
        state_dict_path, weights_only=True, map_location=torch.device("cpu")
    )
    model_cls = argbind.bind(model_cls)
    args = argbind.load_args(Path(state_dict_path).parent / "args.yml")
    if override_args:
        args.update(override_args)
    with argbind.scope(args):
        model = model_cls()
    model.load_state_dict(state_dict)
    if compile:
        model = torch.compile(model)
    return model
