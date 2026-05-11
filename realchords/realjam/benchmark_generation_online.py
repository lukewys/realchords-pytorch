#!/usr/bin/env python
"""Benchmark online decoder generation across PyTorch, ONNX, and MLX."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import seed_everything

from realchords.model.sampling import filter_invalid_tokens_generate_online
from realchords.realjam.agent_interface import (
    ONLINE_MODEL_CANDIDATES,
    _resolve_existing_path,
)
from realchords.realjam.benchmark_generation_mlx import (
    build_aligned_prompt,
    build_sample_prompt,
    get_torch_device,
    load_mlx_model,
    load_pytorch_model,
    load_tokenizer,
    measure_time,
    prepare_onnx_sessions,
)
from realchords.realjam.onnx_kv import generate_tokens_online


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark online generation across PyTorch, ONNX, and MLX"
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
        default="onnx_benchmark_results/generation_online_benchmark.png",
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


def build_online_conditions(
    sample_data: torch.Tensor,
    prompt_length: int,
    condition_length: int,
) -> torch.Tensor:
    trimmed = sample_data.clone()
    if trimmed[0, -1].item() == 2:
        trimmed = trimmed[:, :-1]

    if prompt_length % 2 == 0:
        raise ValueError(
            "Prompt length must be odd so the prompt ends on melody."
        )

    available_conditions = trimmed[:, prompt_length::2]

    if available_conditions.shape[1] == 0:
        last_chord = trimmed[:, -2:-1]
        return last_chord.repeat(1, condition_length)

    if available_conditions.shape[1] >= condition_length:
        return available_conditions[:, :condition_length]

    pad = available_conditions[:, -1:].repeat(
        1, condition_length - available_conditions.shape[1]
    )
    return torch.cat([available_conditions, pad], dim=1)


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
        fmt="s-",
        label="PyTorch model.generate_online",
        capsize=4,
        color="green",
    )
    plt.errorbar(
        input_sizes,
        onnx_means,
        yerr=onnx_errs,
        fmt="s--",
        label="ONNX generate_tokens_online",
        capsize=4,
        color="orange",
    )
    plt.errorbar(
        input_sizes,
        mlx_means,
        yerr=mlx_errs,
        fmt="s-.",
        label="MLX model.generate_online",
        capsize=4,
        color="purple",
    )

    x_values = np.linspace(min(input_sizes), max(input_sizes), 100)
    y_values = x_values / 16
    plt.plot(x_values, y_values, "k--", label="1x realtime at 120 BPM")

    plt.xlabel("Number of generated tokens")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Online Generation Time Comparison: PyTorch vs ONNX vs MLX")
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
    condition_lengths: list[int],
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
                "condition_length": condition_lengths[index],
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

    filter_logits_fn = filter_invalid_tokens_generate_online
    filter_kwargs = {
        "model_part": "melody",
        "tokenizer": tokenizer,
    }

    pytorch_results = []
    onnx_results = []
    mlx_results = []
    prefill_lengths = []
    condition_lengths = []

    for seq_len in input_sizes:
        if seq_len % 2 != 0:
            raise ValueError(
                "input_sizes must be even for online alternating generation"
            )

        max_length = 512 + 1
        prefill_length = max_length - seq_len
        condition_length = seq_len // 2

        prompt = build_aligned_prompt(sample_data, prefill_length)
        conditions = build_online_conditions(
            sample_data,
            prompt.shape[1],
            condition_length,
        )

        prefill_lengths.append(prompt.shape[1])
        condition_lengths.append(conditions.shape[1])

        print(
            f"Testing with generated length {seq_len}, prefill {prompt.shape[1]}, conditions {conditions.shape[1]}"
        )

        prompt_torch = prompt.to(device)
        conditions_torch = conditions.to(device)

        pt_mean, pt_std = measure_time(
            lambda: pytorch_model.generate_online(
                prompts=prompt_torch.clone(),
                conditions=conditions_torch.clone(),
                seq_len=seq_len,
                temperature=args.temperature,
                cache_kv=True,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                reverse_condition_order=True,
            ),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            device=device,
        )
        print(f"  PyTorch: {pt_mean:.4f}s ± {pt_std:.4f}s")
        pytorch_results.append((pt_mean, pt_std))

        onnx_mean, onnx_std = measure_time(
            lambda: generate_tokens_online(
                onnx_init,
                onnx_step,
                prompt.clone(),
                conditions.clone(),
                n_new=seq_len,
                temperature=args.temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                reverse_condition_order=True,
            ),
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            device=torch.device("cpu"),
        )
        print(f"  ONNX:    {onnx_mean:.4f}s ± {onnx_std:.4f}s")
        onnx_results.append((onnx_mean, onnx_std))

        mlx_mean, mlx_std = measure_time(
            lambda: mlx_model.generate_online(
                prompts=prompt.clone(),
                conditions=conditions.clone(),
                seq_len=seq_len,
                temperature=args.temperature,
                cache_kv=True,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                reverse_condition_order=True,
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
        condition_lengths,
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
