#!/usr/bin/env python
"""Benchmark decoder generation across PyTorch, ONNX, and MLX backends."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything

from realchords.constants import REALJAM_CHECKPOINT_DIR
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.model.gen_model import DecoderTransformer
from realchords.model.sampling import (
    filter_invalid_tokens_generate_unconditional,
)
from realchords.realjam.agent_interface import (
    ONLINE_MODEL_CANDIDATES,
    _resolve_existing_path,
)
from realchords.realjam.mlx_models import MLXDecoderTransformer
from realchords.realjam.onnx_kv import generate_tokens
from realchords.realjam.prepare_onnx import prepare_online
from realchords.utils.inference_utils import load_gen_model_from_state_dict

SAMPLE_DATA = """1  2099  187  748   59  748   59  748   59  748   59  748   59 1877  185
526   57  526   57  526   57  526  190  526   62  526   62  526   62
1656  183  305   55 1656   55  305   55  305   55  305   55  305  183
305   55  305  183  305   55  305  188  305   60  305  188  305   60
305  187  305   59  305   59  305   59 1656  185  305   57  305   57
305   57  305   57  305   57 1877  183  526   55  526  188  526   60
526  188  526   60  526  187  526   59 2099   59  748   59 2099  185
748   57  748   57  748   57  748   57  748   57  748  183  748   55
748   55  748   55  748   55  748   55  748    3  748    3  748    3
748    3 2099  187  748   59  748   59  748   59  748  185  748   57
2471  185 1120   57 1120   57 1120   57 1120  190 1120   62 1120   62
1120   62 1656  183  305   55 1656   55  305  182  305  180  305   52
305   52  305   52  305  188  305   60  305   60  305   60  305  187
305   59  305   59  305   59  305  187  305   59 1656  185  305   57
305   57  305   57  305   57  305   57 1877  188  526   60  526   60
526   60  526  188  526   60  526   60  526   60 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57  748   57  748   57  748  183
748   55    3   55    3   55    3   55    3   55 2099  187  748   59
2099  185  748   57  748   57  748   57    2"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark decoder generation across PyTorch, ONNX, and MLX"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=_resolve_existing_path(ONLINE_MODEL_CANDIDATES),
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="onnx_benchmark_results/generation_backend_benchmark.png",
        help="Path to save the comparison figure",
    )
    parser.add_argument(
        "--input_sizes",
        type=str,
        default="8,16,24,32",
        help="Comma-separated list of generated token counts",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats per input size",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=1,
        help="Warmup runs excluded from timing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (defaults to cuda if available, else cpu)",
    )
    parser.add_argument(
        "--onnx_provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider",
    )
    return parser.parse_args()


def load_tokenizer() -> HooktheoryTokenizer:
    chord_names_path = os.path.join(
        REALJAM_CHECKPOINT_DIR, "chord_names_augmented.json"
    )
    with open(chord_names_path, "r") as handle:
        chord_names = json.load(handle)
    return HooktheoryTokenizer(chord_names=chord_names)


def build_sample_prompt() -> torch.Tensor:
    sample = [int(token) for token in SAMPLE_DATA.split()]
    return torch.tensor(sample, dtype=torch.long).unsqueeze(0)


def build_aligned_prompt(
    sample_data: torch.Tensor, target_length: int
) -> torch.Tensor:
    prompt = sample_data.clone()

    if prompt[0, -1].item() == 2:
        prompt = prompt[:, :-1]

    if prompt.shape[1] >= target_length:
        return prompt[:, :target_length]

    pad_count = target_length - prompt.shape[1]
    if pad_count % 2 != 0:
        raise ValueError(
            "Prefill padding must preserve chord/melody alternation; use even seq_len."
        )

    last_pair = prompt[:, -2:]
    repeated_pairs = last_pair.repeat(1, pad_count // 2)
    return torch.cat([prompt, repeated_pairs], dim=1)


def get_torch_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def measure_time(
    run_fn,
    *,
    repeats: int,
    warmup_runs: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup_runs):
        run_fn()
        maybe_synchronize(device)

    times = []
    for _ in range(repeats):
        maybe_synchronize(device)
        start = time.perf_counter()
        run_fn()
        maybe_synchronize(device)
        times.append(time.perf_counter() - start)

    return float(np.mean(times)), float(np.std(times))


def load_pytorch_model(
    model_path: str,
    tokenizer: HooktheoryTokenizer,
    device: torch.device,
) -> DecoderTransformer:
    model = load_gen_model_from_state_dict(
        model_path,
        DecoderTransformer,
        compile=False,
        override_args={
            "DecoderTransformer.num_tokens": tokenizer.num_tokens,
            "DecoderTransformer.pad_value": tokenizer.pad_token,
        },
    )
    model.eval()
    model.to(device)
    return model


def load_mlx_model(
    model_path: str,
    tokenizer: HooktheoryTokenizer,
) -> MLXDecoderTransformer:
    return MLXDecoderTransformer.from_checkpoint(
        model_path,
        tokenizer,
        override_args={
            "DecoderTransformer.num_tokens": tokenizer.num_tokens,
            "DecoderTransformer.pad_value": tokenizer.pad_token,
        },
    )


def prepare_onnx_sessions(
    model: DecoderTransformer,
    tokenizer: HooktheoryTokenizer,
    model_path: str,
    provider: str,
):
    save_dir = os.path.join(os.path.dirname(model_path), "onnx")
    return prepare_online(
        model,
        tokenizer,
        save_dir=save_dir,
        model_name="onnx_online",
        max_gen_seq_len=512,
        provider=provider,
    )


def create_comparison_plot(
    input_sizes: list[int],
    pytorch_results: list[tuple[float, float]],
    onnx_results: list[tuple[float, float]],
    mlx_results: list[tuple[float, float]],
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 6))

    pytorch_means = [result[0] for result in pytorch_results]
    onnx_means = [result[0] for result in onnx_results]
    mlx_means = [result[0] for result in mlx_results]

    pytorch_errs = [result[1] * 2 for result in pytorch_results]
    onnx_errs = [result[1] * 2 for result in onnx_results]
    mlx_errs = [result[1] * 2 for result in mlx_results]

    plt.errorbar(
        input_sizes,
        pytorch_means,
        yerr=pytorch_errs,
        fmt="o-",
        label="PyTorch model.generate",
        capsize=4,
        color="blue",
    )
    plt.errorbar(
        input_sizes,
        onnx_means,
        yerr=onnx_errs,
        fmt="o--",
        label="ONNX generate_tokens",
        capsize=4,
        color="red",
    )
    plt.errorbar(
        input_sizes,
        mlx_means,
        yerr=mlx_errs,
        fmt="o-.",
        label="MLX model.generate",
        capsize=4,
        color="green",
    )

    x_values = np.linspace(min(input_sizes), max(input_sizes), 100)
    y_values = x_values / 16
    plt.plot(x_values, y_values, "k--", label="1x realtime at 120 BPM")

    plt.xlabel("Number of generated tokens")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Generation Time Comparison: PyTorch vs ONNX vs MLX")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to {output_path}")


def save_summary_json(
    output_path: str,
    model_path: str,
    device: torch.device,
    onnx_provider: str,
    input_sizes: list[int],
    prefill_lengths: list[int],
    pytorch_results: list[tuple[float, float]],
    onnx_results: list[tuple[float, float]],
    mlx_results: list[tuple[float, float]],
    repeats: int,
    warmup_runs: int,
) -> str:
    output_json = str(Path(output_path).with_suffix(".json"))
    summary = {
        "model_path": model_path,
        "pytorch_device": str(device),
        "onnx_provider": onnx_provider,
        "repeats": repeats,
        "warmup_runs": warmup_runs,
        "results": [],
    }

    for index, seq_len in enumerate(input_sizes):
        pt_mean, pt_std = pytorch_results[index]
        onnx_mean, onnx_std = onnx_results[index]
        mlx_mean, mlx_std = mlx_results[index]
        summary["results"].append(
            {
                "generated_tokens": seq_len,
                "prefill_length": prefill_lengths[index],
                "pytorch_mean_s": pt_mean,
                "pytorch_std_s": pt_std,
                "onnx_mean_s": onnx_mean,
                "onnx_std_s": onnx_std,
                "mlx_mean_s": mlx_mean,
                "mlx_std_s": mlx_std,
                "onnx_speedup_vs_pytorch": pt_mean / onnx_mean,
                "mlx_speedup_vs_pytorch": pt_mean / mlx_mean,
                "mlx_speedup_vs_onnx": onnx_mean / mlx_mean,
            }
        )

    with open(output_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Summary saved to {output_json}")
    return output_json


def main() -> None:
    args = parse_args()
    seed_everything(42)

    input_sizes = [int(size) for size in args.input_sizes.split(",")]
    device = get_torch_device(args.device)
    model_path = args.model_path

    print(f"Using PyTorch device: {device}")
    print(f"Using ONNX provider: {args.onnx_provider}")
    print(f"Using checkpoint: {model_path}")

    tokenizer = load_tokenizer()
    sample_data = build_sample_prompt()

    pytorch_model = load_pytorch_model(model_path, tokenizer, device)
    mlx_model = load_mlx_model(model_path, tokenizer)
    onnx_init, onnx_step = prepare_onnx_sessions(
        pytorch_model,
        tokenizer,
        model_path,
        args.onnx_provider,
    )

    filter_logits_fn = filter_invalid_tokens_generate_unconditional
    filter_kwargs = {
        "model_part": "chord",
        "tokenizer": tokenizer,
        "filter_opposite_part": True,
    }

    pytorch_results = []
    onnx_results = []
    mlx_results = []
    prefill_lengths = []

    for seq_len in input_sizes:
        if seq_len % 2 != 0:
            raise ValueError(
                "input_sizes must be even for alternating chord generation"
            )

        max_length = 512 + 1
        prefill_length = max_length - seq_len
        prompt = build_aligned_prompt(sample_data, prefill_length)
        prefill_lengths.append(prompt.shape[1])

        print(
            f"Testing with generated length {seq_len} and prefill {prompt.shape[1]}"
        )

        prompt_torch = prompt.to(device)

        pt_mean, pt_std = measure_time(
            lambda: pytorch_model.generate(
                prompts=prompt_torch.clone(),
                seq_len=seq_len,
                temperature=args.temperature,
                cache_kv=True,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
            ),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            device=device,
        )
        print(f"  PyTorch: {pt_mean:.4f}s ± {pt_std:.4f}s")
        pytorch_results.append((pt_mean, pt_std))

        onnx_mean, onnx_std = measure_time(
            lambda: generate_tokens(
                onnx_init,
                onnx_step,
                prompt.clone(),
                n_new=seq_len,
                temperature=args.temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
            ),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            device=torch.device("cpu"),
        )
        print(f"  ONNX:    {onnx_mean:.4f}s ± {onnx_std:.4f}s")
        onnx_results.append((onnx_mean, onnx_std))

        mlx_mean, mlx_std = measure_time(
            lambda: mlx_model.generate(
                prompts=prompt.clone(),
                seq_len=seq_len,
                temperature=args.temperature,
                cache_kv=True,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
            ),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            device=torch.device("cpu"),
        )
        print(f"  MLX:     {mlx_mean:.4f}s ± {mlx_std:.4f}s")
        mlx_results.append((mlx_mean, mlx_std))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    create_comparison_plot(
        input_sizes,
        pytorch_results,
        onnx_results,
        mlx_results,
        args.output_path,
    )

    save_summary_json(
        args.output_path,
        model_path,
        device,
        args.onnx_provider,
        input_sizes,
        prefill_lengths,
        pytorch_results,
        onnx_results,
        mlx_results,
        args.repeats,
        args.warmup_runs,
    )

    print("\nSpeed Comparisons:")
    for index, seq_len in enumerate(input_sizes):
        pt_time = pytorch_results[index][0]
        onnx_time = onnx_results[index][0]
        mlx_time = mlx_results[index][0]
        print(f"Sequence length {seq_len}:")
        print(f"  ONNX vs PyTorch: {pt_time / onnx_time:.2f}x")
        print(f"  MLX  vs PyTorch: {pt_time / mlx_time:.2f}x")
        print(f"  MLX  vs ONNX:    {onnx_time / mlx_time:.2f}x")


if __name__ == "__main__":
    main()
