"""CPU and CUDA GPU inference latency, throughput, and VRAM benchmarking (Phase 5).

Performs warm-up iterations and strict `torch.cuda.synchronize()` barriers when
measuring GPU latency, reporting mean, median, p95 latency (ms), throughput
(images/sec), peak GPU VRAM, batch size, precision mode, and image dimensions.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def benchmark_model_inference(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 1,
    precision: str = "fp32",
    warmup_runs: int = 10,
    measured_runs: int = 40,
) -> Dict[str, Any]:
    """Measure latency statistics and throughput on the specified device and precision."""
    is_cuda = device.type == "cuda"
    use_amp_fp16 = is_cuda and precision.lower() == "fp16"

    model = model.to(device)
    model.eval()
    dummy = torch.randn((batch_size, *input_size), dtype=torch.float32, device=device)

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up passes (excluded from timing)
    with torch.no_grad():
        for _ in range(warmup_runs):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp_fp16):
                _ = model(dummy)
        if is_cuda:
            torch.cuda.synchronize(device)

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(measured_runs):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp_fp16):
                _ = model(dummy)
            if is_cuda:
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    p95_ms = float(np.percentile(arr, 95))
    throughput_ips = (batch_size * 1000.0) / max(mean_ms, 1e-6)

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if is_cuda else None
    )
    hw_name = (
        torch.cuda.get_device_name(device)
        if is_cuda
        else (platform.processor() or platform.machine() or "CPU")
    )

    return {
        "device": str(device),
        "hardware_name": hw_name,
        "precision": "amp_fp16" if use_amp_fp16 else "fp32",
        "batch_size": int(batch_size),
        "input_dimensions": list(input_size),
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "mean_latency_ms": round(mean_ms, 3),
        "median_latency_ms": round(median_ms, 3),
        "p95_latency_ms": round(p95_ms, 3),
        "throughput_images_per_sec": round(throughput_ips, 2),
        "peak_gpu_memory_mb": peak_vram_mb,
    }


def run_full_hardware_benchmark(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    cpu_runs: int = 20,
    gpu_runs: int = 50,
) -> Dict[str, Any]:
    """Run standardized CPU (bs=1 FP32) and CUDA GPU (bs=1 FP32, bs=16 FP32 & FP16) benchmarks."""
    results: Dict[str, Any] = {}
    results["cpu_bs1_fp32"] = benchmark_model_inference(
        model=model,
        device=torch.device("cpu"),
        input_size=input_size,
        batch_size=1,
        precision="fp32",
        warmup_runs=5,
        measured_runs=cpu_runs,
    )

    if torch.cuda.is_available():
        gpu_dev = torch.device("cuda:0")
        results["gpu_bs1_fp32"] = benchmark_model_inference(
            model=model,
            device=gpu_dev,
            input_size=input_size,
            batch_size=1,
            precision="fp32",
            warmup_runs=10,
            measured_runs=gpu_runs,
        )
        results["gpu_bs16_fp32"] = benchmark_model_inference(
            model=model,
            device=gpu_dev,
            input_size=input_size,
            batch_size=16,
            precision="fp32",
            warmup_runs=10,
            measured_runs=gpu_runs,
        )
        results["gpu_bs16_fp16"] = benchmark_model_inference(
            model=model,
            device=gpu_dev,
            input_size=input_size,
            batch_size=16,
            precision="fp16",
            warmup_runs=10,
            measured_runs=gpu_runs,
        )

    return results
