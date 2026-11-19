"""GPU and PyTorch CUDA verification test for PlantDiseaseAI (Phase 1).

Can be executed either directly or via pytest:
    python ml/tests/test_gpu.py
    pytest ml/tests/test_gpu.py -v
"""

from __future__ import annotations

import sys
import time

import torch


def verify_gpu(matrix_size: int = 1024) -> dict:
    """Verify PyTorch installation, CUDA availability, and GPU tensor computation.

    Returns a structured diagnostic dictionary describing the execution result.
    """
    info: dict = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": 0,
        "gpu_name": None,
        "compute_capability": None,
        "total_vram_gb": None,
        "matmul_verified": False,
        "elapsed_ms": None,
        "device_used": "cpu",
    }

    print("=" * 68)
    print("PlantDiseaseAI — PyTorch & NVIDIA GPU Verification")
    print("=" * 68)
    print(f"Python Version:     {info['python_version']}")
    print(f"PyTorch Version:    {info['torch_version']}")
    print(f"PyTorch CUDA Build: {info['torch_cuda_version'] or 'None (CPU-only build)'}")
    print(f"CUDA Available:     {info['cuda_available']}")

    if not info["cuda_available"]:
        print("\n[WARNING] CUDA GPU support is NOT available in this PyTorch environment.")
        print("Common causes:")
        print("  1. CPU-only PyTorch wheel is installed instead of a CUDA wheel.")
        print("  2. NVIDIA GPU driver is missing or outdated for this CUDA build.")
        print("  3. GPU compute capability requires a newer CUDA wheel (e.g., cu128 for RTX 50-series).")
        print("\nRecommended fix (official PyTorch CUDA 12.8 wheel):")
        print("  uv pip install --python .venv\\Scripts\\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("\nRunning safe CPU fallback matrix multiplication check...")

        a_cpu = torch.randn((256, 256), dtype=torch.float32, device="cpu")
        b_cpu = torch.randn((256, 256), dtype=torch.float32, device="cpu")
        t0 = time.perf_counter()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert c_cpu.shape == (256, 256) and torch.isfinite(c_cpu).all().item()
        info["matmul_verified"] = True
        info["elapsed_ms"] = round(elapsed_ms, 2)
        info["device_used"] = "cpu"
        print(f"[CPU FALLBACK OK] 256x256 matmul succeeded on CPU in {elapsed_ms:.2f} ms.")
        print("=" * 68)
        return info

    # GPU is available — inspect device properties
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    info["gpu_count"] = torch.cuda.device_count()
    info["gpu_name"] = torch.cuda.get_device_name(device)
    info["compute_capability"] = f"{props.major}.{props.minor}"
    info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)
    info["device_used"] = str(device)

    print(f"Detected GPU Count: {info['gpu_count']}")
    print(f"Detected GPU Name:  {info['gpu_name']}")
    print(f"Compute Capability: sm_{props.major}{props.minor} ({info['compute_capability']})")
    print(f"Total GPU Memory:   {info['total_vram_gb']:.2f} GB")

    # Allocate tensors on GPU and run matrix multiplication
    print(f"\nAllocating ({matrix_size} x {matrix_size}) float32 tensors on {device}...")
    a = torch.randn((matrix_size, matrix_size), dtype=torch.float32, device=device)
    b = torch.randn((matrix_size, matrix_size), dtype=torch.float32, device=device)

    # Warmup + timed synchronized matmul
    _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    c = torch.matmul(a, b)
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Numerical sanity verification
    is_finite = bool(torch.isfinite(c).all().item())
    assert c.device.type == "cuda", "Result tensor is not on CUDA device."
    assert c.shape == (matrix_size, matrix_size), f"Unexpected output shape: {c.shape}"
    assert is_finite, "GPU matrix multiplication produced non-finite (NaN/Inf) values."

    info["matmul_verified"] = True
    info["elapsed_ms"] = round(elapsed_ms, 2)

    print(f"Result Tensor:      shape={tuple(c.shape)}, dtype={c.dtype}, device={c.device}")
    print(f"Sample Value:       c[0, 0] = {c[0, 0].item():.4f} (finite={is_finite})")
    print(f"Execution Time:     {elapsed_ms:.2f} ms")
    print("[SUCCESS] GPU tensor allocation and matrix multiplication verified!")
    print("=" * 68)
    return info


def test_pytorch_cuda_gpu() -> None:
    """Pytest test asserting CUDA GPU availability and successful GPU matmul."""
    result = verify_gpu(matrix_size=1024)
    assert result["cuda_available"] is True, (
        "Expected CUDA GPU to be available, but torch.cuda.is_available() returned False."
    )
    assert result["gpu_name"], "Expected a detected NVIDIA GPU name."
    assert result["matmul_verified"] is True, "GPU matrix multiplication failed verification."


if __name__ == "__main__":
    summary = verify_gpu(matrix_size=1024)
    sys.exit(0 if summary["matmul_verified"] else 1)
