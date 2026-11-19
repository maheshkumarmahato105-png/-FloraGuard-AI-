"""Automated architecture, forward/backward, CPU/GPU, and checkpoint tests (Phase 3)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from models import (
    SUPPORTED_ARCHITECTURES,
    AgriVisionCNN,
    MultiScaleFeatureBlock,
    ResidualBlock,
    SqueezeExcitation,
    create_model,
    load_model_checkpoint,
    load_model_config,
    resolve_verified_classes,
    save_model_checkpoint,
)


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_model_init_forward_backward_cpu_with_verified_dataset_classes(arch: str):
    """Verify initialization, (4, 3, 224, 224) forward pass, raw logits, backprop,
    gradient availability, and zero NaN/Inf on CPU using verified Phase 2 class mapping."""
    cfg = load_model_config()
    class_names, _ = resolve_verified_classes(cfg)
    expected_classes = len(class_names)
    assert expected_classes >= 2

    model = create_model(architecture=arch, model_config=cfg, seed=42)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 0
    assert trainable_params == total_params

    x = torch.randn(4, 3, 224, 224, dtype=torch.float32)
    targets = torch.randint(0, expected_classes, (4,), dtype=torch.long)

    logits = model(x)
    assert logits.shape == (4, expected_classes)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all().item(), f"Non-finite logits in {arch}"

    # Verify model outputs raw logits (NOT softmax probabilities summing to 1 with all >= 0)
    row_sums = logits.sum(dim=1)
    looks_like_softmax = bool((logits >= 0).all().item() and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))
    assert not looks_like_softmax, f"{arch} appears to apply softmax inside forward()!"

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    assert torch.isfinite(loss).item(), f"Non-finite loss in {arch}"
    loss.backward()

    missing_grads = [
        name for name, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert len(missing_grads) == 0, f"Parameters missing gradients in {arch}: {missing_grads}"

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item(), f"Non-finite gradient in {arch}.{name}"


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_dynamic_synthetic_class_count_and_custom_input_resolution(arch: str):
    """Verify dynamic class count works cleanly with a small synthetic configuration."""
    synthetic_classes = 5
    cfg = load_model_config()
    model = create_model(
        architecture=arch,
        model_config=cfg,
        num_classes=synthetic_classes,
        seed=7,
    )
    model.eval()
    x = torch.randn(4, 3, 128, 128, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, synthetic_classes)
    assert torch.isfinite(out).all().item()


def test_agrivision_submodules_and_projection_shortcuts():
    """Unit-test SqueezeExcitation, MultiScaleFeatureBlock, and ResidualBlock independently."""
    se = SqueezeExcitation(channels=64, reduction_ratio=8)
    x_se = torch.randn(4, 64, 28, 28)
    y_se = se(x_se)
    assert y_se.shape == x_se.shape and torch.isfinite(y_se).all().item()

    ms = MultiScaleFeatureBlock(in_channels=32, branch_3x3_channels=32, branch_5x5_channels=32, out_channels=64)
    x_ms = torch.randn(4, 32, 112, 112)
    y_ms = ms(x_ms)
    assert y_ms.shape == (4, 64, 112, 112) and torch.isfinite(y_ms).all().item()

    # Projection shortcut when channel count & stride change
    res_proj = ResidualBlock(in_channels=64, out_channels=128, stride=2, use_attention=True)
    assert not isinstance(res_proj.shortcut, nn.Identity)
    y_proj = res_proj(torch.randn(4, 64, 56, 56))
    assert y_proj.shape == (4, 128, 28, 28)

    # Identity shortcut when dimensions match
    res_id = ResidualBlock(in_channels=128, out_channels=128, stride=1, use_attention=True)
    assert isinstance(res_id.shortcut, nn.Identity)
    y_id = res_id(y_proj)
    assert y_id.shape == (4, 128, 28, 28)


def test_reproducible_initialization_and_input_validation():
    """Same seed produces identical initial weights; invalid input shapes raise ValueError."""
    cfg = load_model_config()
    m1 = create_model(architecture="agrivision_cnn", model_config=cfg, num_classes=6, seed=123)
    m2 = create_model(architecture="agrivision_cnn", model_config=cfg, num_classes=6, seed=123)

    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Initialization mismatch on {n1}"

    with pytest.raises(ValueError):
        m1(torch.randn(3, 224, 224))  # 3D instead of 4D
    with pytest.raises(ValueError):
        m1(torch.randn(4, 1, 224, 224))  # Wrong channel count


@pytest.mark.parametrize("arch", SUPPORTED_ARCHITECTURES)
def test_gpu_forward_and_backward_when_cuda_available(arch: str):
    """Verify full forward and backward execution on NVIDIA CUDA GPU when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available in this environment.")

    device = torch.device("cuda:0")
    cfg = load_model_config()
    class_names, _ = resolve_verified_classes(cfg)
    num_classes = len(class_names)

    model = create_model(architecture=arch, model_config=cfg, seed=42).to(device)
    model.train()

    x = torch.randn(4, 3, 224, 224, dtype=torch.float32, device=device)
    targets = torch.randint(0, num_classes, (4,), dtype=torch.long, device=device)

    logits = model(x)
    assert logits.device.type == "cuda"
    assert logits.shape == (4, num_classes)
    assert torch.isfinite(logits).all().item()

    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    torch.cuda.synchronize(device)

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.device.type == "cuda"
            assert torch.isfinite(p.grad).all().item()


def test_checkpoint_save_and_load_roundtrip_in_tmp_dir(tmp_path: Path):
    """Verify reusable checkpoint utilities without creating fake trained checkpoints in artifacts/."""
    cfg = load_model_config()
    class_names = ["class_a", "class_b", "class_c", "class_d"]
    model = create_model(architecture="agrivision_cnn", model_config=cfg, num_classes=4, seed=42)

    ckpt_file = tmp_path / "unit_test_init_only.pt"
    save_model_checkpoint(
        filepath=ckpt_file,
        model=model,
        architecture_name="agrivision_cnn",
        model_config=cfg,
        class_names=class_names,
        random_seed=42,
        class_mapping={"classes": class_names},
    )

    loaded = load_model_checkpoint(ckpt_file, map_location="cpu")
    assert loaded["architecture_name"] == "agrivision_cnn"
    assert loaded["random_seed"] == 42
    assert loaded["class_names"] == class_names
    assert loaded["num_classes"] == 4

    restored = create_model(architecture="agrivision_cnn", model_config=cfg, num_classes=4, seed=99)
    restored.load_state_dict(loaded["model_state_dict"])

    model.eval()
    restored.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x), atol=1e-6)
