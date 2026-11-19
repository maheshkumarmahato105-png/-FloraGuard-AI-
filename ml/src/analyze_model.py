"""Architecture Profiling, Comparison, Diagram Generation & Report Builder (Phase 3).

Profiles all three Phase 3 architectures using actual PyTorch forward hooks on
a real `(1, 3, 224, 224)` tensor:
  1. Simple CNN (`simple_cnn`)
  2. AgriVision-CNN (`agrivision_cnn`)
  3. ResNet18 Baseline (`resnet18_baseline`, from scratch, weights=None)

Computes:
  - Total parameters & trainable parameters
  - Approximate FP32 and FP16 model size (MB)
  - Exact Conv2d / Linear MACs & FLOPs (`FLOPs = 2 * MACs`)
  - Layer-by-layer input/output tensor dimensions for AgriVision-CNN

Generates:
  - `ml/reports/charts/agrivision_cnn_architecture.png`
  - `ml/reports/model_architecture_report.md`

Does NOT train models or fabricate accuracy metrics.

Usage:
    python ml/src/analyze_model.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import (
    SUPPORTED_ARCHITECTURES,
    create_model,
    load_model_config,
    resolve_verified_classes,
)


def _format_shape(tensor: Any) -> str:
    if isinstance(tensor, torch.Tensor):
        return "×".join(str(d) for d in tensor.shape)
    if isinstance(tensor, (tuple, list)) and tensor and isinstance(tensor[0], torch.Tensor):
        return "×".join(str(d) for d in tensor[0].shape)
    return "—"


def profile_model(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
) -> Dict[str, Any]:
    """Profile exact parameter counts, memory footprint, layer shapes, and MACs/FLOPs."""
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    buffer_elements = sum(b.numel() for b in model.buffers())

    # Approximate serialized state_dict weight sizes
    fp32_mb = ((total_params + buffer_elements) * 4) / (1024**2)
    fp16_mb = ((total_params + buffer_elements) * 2) / (1024**2)

    layer_rows: List[Dict[str, Any]] = []
    total_macs = 0
    hooks = []

    profiled_types = (
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm2d,
        nn.MaxPool2d,
        nn.AdaptiveAvgPool2d,
        nn.Dropout,
    )

    def make_hook(layer_name: str):
        def _hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            nonlocal total_macs
            macs = 0
            if isinstance(module, nn.Conv2d) and isinstance(output, torch.Tensor):
                out_h, out_w = output.shape[2], output.shape[3]
                k_h, k_w = module.kernel_size
                # MACs per output pixel = (in_channels / groups) * k_h * k_w
                macs = int(
                    output.shape[0]
                    * out_h
                    * out_w
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * k_h
                    * k_w
                )
            elif isinstance(module, nn.Linear) and isinstance(output, torch.Tensor):
                macs = int(output.shape[0] * module.in_features * module.out_features)
            elif isinstance(module, nn.BatchNorm2d) and isinstance(output, torch.Tensor):
                # Scale + shift elementwise ops (counted as ~2 ops -> 1 MAC per element)
                macs = int(output.numel())

            total_macs += macs
            params = sum(p.numel() for p in module.parameters(recurse=False))
            layer_rows.append(
                {
                    "name": layer_name,
                    "type": module.__class__.__name__,
                    "input_shape": _format_shape(inputs),
                    "output_shape": _format_shape(output),
                    "params": int(params),
                    "macs": int(macs),
                }
            )

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, profiled_types):
            hooks.append(module.register_forward_hook(make_hook(name)))

    dummy = torch.randn(input_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(dummy)

    for h in hooks:
        h.remove()

    total_flops = total_macs * 2
    return {
        "input_shape": list(input_size),
        "output_shape": list(out.shape),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "non_trainable_params": int(non_trainable_params),
        "fp32_size_mb": round(fp32_mb, 3),
        "fp16_size_mb": round(fp16_mb, 3),
        "total_macs": int(total_macs),
        "total_gmacs": round(total_macs / 1e9, 4),
        "total_flops": int(total_flops),
        "total_gflops": round(total_flops / 1e9, 4),
        "num_profiled_layers": len(layer_rows),
        "layers": layer_rows,
    }


def draw_agrivision_diagram(
    num_classes: int,
    av_stats: Dict[str, Any],
    out_path: Path,
) -> None:
    """Render a crisp publication-style block diagram of AgriVision-CNN."""
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, fc, ec, title_color="#0f172a"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#334155",
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.6),
        )

    ax.text(
        7.0,
        7.2,
        f"AgriVision-CNN Architecture Diagram  |  {av_stats['total_params']:,} Params  |  "
        f"{av_stats['total_gflops']:.2f} GFLOPs  |  {av_stats['fp32_size_mb']:.2f} MB (FP32)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#065f46",
    )

    # Top row: Input -> Stem -> Multi-Scale (parallel 3x3 & 5x5 -> Concat + 1x1 Proj)
    box(0.4, 5.1, 2.0, 1.25, "Input Image", "224 × 224 × 3\n(RGB Leaf)", "#f8fafc", "#64748b")
    box(3.0, 5.1, 2.3, 1.25, "Conv Stem", "3×3 Conv (s=2) → 32\nBN + ReLU\n112 × 112 × 32", "#ecfdf5", "#059669")

    # Parallel Multi-Scale branches
    box(6.0, 5.85, 2.3, 0.9, "Branch 1 (Fine)", "3×3 Conv → 32ch\nSmall Disease Spots", "#e0f2fe", "#0284c7")
    box(6.0, 4.65, 2.3, 0.9, "Branch 2 (Context)", "5×5 Conv → 32ch\nHalos & Blotches", "#e0f2fe", "#0284c7")
    box(9.0, 5.1, 2.3, 1.25, "Multi-Scale Fusion", "Concat (64ch) + BN + ReLU\n1×1 Projection → 64ch\n112 × 112 × 64", "#dcfce7", "#16a34a")

    arrow(2.52, 5.72, 2.88, 5.72)
    arrow(5.42, 5.95, 5.88, 6.25)
    arrow(5.42, 5.50, 5.88, 5.10)
    arrow(8.42, 6.25, 8.88, 5.95)
    arrow(8.42, 5.10, 8.88, 5.50)

    # Bottom row: Residual+SE Stages (64 -> 128 -> 256) -> Head
    box(1.0, 1.65, 2.6, 1.65, "Residual + SE Stage 1", "2 × ResidualBlock (s=2,1)\n64 ch · SE (r=8)\nProj Shortcut\n56 × 56 × 64", "#fef3c7", "#d97706")
    box(4.2, 1.65, 2.6, 1.65, "Residual + SE Stage 2", "2 × ResidualBlock (s=2,1)\n128 ch · SE (r=8)\nProj Shortcut\n28 × 28 × 128", "#fef3c7", "#d97706")
    box(7.4, 1.65, 2.6, 1.65, "Residual + SE Stage 3", "2 × ResidualBlock (s=2,1)\n256 ch · SE (r=8)\nProj Shortcut\n14 × 14 × 256", "#fef3c7", "#d97706")
    box(10.6, 1.65, 2.8, 1.65, "Classification Head", f"Global Avg Pool (256)\nDropout (p=0.30)\nLinear → {num_classes} Logits\n(No Softmax in forward)", "#ede9fe", "#7c3aed")

    # Connector from Multi-Scale Fusion down to Stage 1
    ax.plot([10.15, 10.15, 2.3, 2.3], [4.98, 4.05, 4.05, 3.45], color="#475569", lw=1.6)
    arrow(2.3, 3.60, 2.3, 3.42)

    arrow(3.72, 2.48, 4.08, 2.48)
    arrow(6.92, 2.48, 7.28, 2.48)
    arrow(10.12, 2.48, 10.48, 2.48)

    # SE Attention inset note
    box(
        3.2,
        0.25,
        7.6,
        0.92,
        "Inside Each Residual Block: Squeeze-and-Excitation (SE) Channel Attention",
        "Conv 3×3 → BN → ReLU → Conv 3×3 → BN → [GAP → 1×1 Conv (C/8) → ReLU → 1×1 Conv (C) → Sigmoid → Scale] + Shortcut → ReLU",
        "#f1f5f9",
        "#94a3b8",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_architecture_report(
    config: Dict[str, Any],
    class_names: List[str],
    profiles: Dict[str, Dict[str, Any]],
    report_path: Path,
    diagram_rel_path: str,
) -> None:
    """Write ml/reports/model_architecture_report.md with exact profiled numbers."""
    av = profiles["agrivision_cnn"]
    sc = profiles["simple_cnn"]
    rn = profiles["resnet18_baseline"]
    h = config["input"]["image_height"]
    w = config["input"]["image_width"]
    c = config["input"]["in_channels"]

    lines = [
        "# PlantDiseaseAI — Phase 3 Model Architecture & Computational Analysis Report",
        "",
        f"- **Generated At (UTC):** `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Input Tensor Shape:** `(B, {c}, {h}, {w})` (profiled at batch size `1` for MACs/FLOPs)",
        f"- **Dynamic Output Classes:** `{len(class_names)}` (resolved automatically from verified Phase 2 `ml/configs/class_mapping.yaml`)",
        f"- **Random Seed:** `{config['experiment']['random_seed']}`",
        "- **Training Status:** **Phase 3 Architecture Development Only — No model training has been performed, and no accuracy claims are made.**",
        "",
        "---",
        "",
        "## 1. Proposed Architecture: AgriVision-CNN",
        "",
        f"Visual diagram: `{diagram_rel_path}`",
        "",
        "AgriVision-CNN is a custom, mobile-conscious multi-scale residual convolutional architecture tailored for plant leaf pathology:",
        "",
        "1. **Fine-Grained Initial Stem (`3×3`, 32 channels, stride 2, BN, ReLU):** Uses a `3×3` kernel instead of ResNet's aggressive `7×7` + MaxPool stem to preserve small foliar lesions, pinpoint fungal pycnidia, and early necrotic edges.",
        "2. **Multi-Scale Feature Extraction Block (Parallel `3×3` & `5×5` Branches → Concat → BN → ReLU → `1×1` Projection):** Simultaneously captures fine disease spots (`3×3`) and wider yellow chlorosis halos / blight blotches (`5×5`), fusing cross-scale maps into 64 channels.",
        "3. **Residual Learning Stages (`64 → 128 → 256` channels, 2 blocks/stage):** Deep residual blocks with `1×1` projection shortcuts whenever channel depth or spatial resolution changes, preventing vanishing gradients during training from scratch.",
        "4. **Squeeze-and-Excitation (SE) Channel Attention (`r=8`):** Lightweight GAP → `1×1` Conv → ReLU → `1×1` Conv → Sigmoid channel reweighting embedded inside every residual block to emphasize disease-discriminative color/texture channels while suppressing background clutter (important for PlantDoc generalization).",
        "5. **Classification Head (`GlobalAvgPool2d → Dropout(0.30) → Linear`):** Resolution-flexible pooling returning **raw logits** (no softmax in `forward()`), keeping memory footprint low for future mobile deployment.",
        "",
        "---",
        "",
        "## 2. Three-Architecture Comparison (Measured Structural & Compute Metrics)",
        "",
        "| Architecture | Total Params | Trainable Params | FP32 Size (MB) | FP16 Size (MB) | GMACs (1×3×224×224) | GFLOPs (2×MACs) | Pretrained Weights |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
        f"| **Simple CNN (`simple_cnn`)** | {sc['total_params']:,} | {sc['trainable_params']:,} | {sc['fp32_size_mb']:.2f} | {sc['fp16_size_mb']:.2f} | {sc['total_gmacs']:.3f} | {sc['total_gflops']:.3f} | None (from scratch) |",
        f"| **AgriVision-CNN (`agrivision_cnn`)** | {av['total_params']:,} | {av['trainable_params']:,} | {av['fp32_size_mb']:.2f} | {av['fp16_size_mb']:.2f} | {av['total_gmacs']:.3f} | {av['total_gflops']:.3f} | None (from scratch) |",
        f"| **ResNet18 Baseline (`resnet18_baseline`)** | {rn['total_params']:,} | {rn['trainable_params']:,} | {rn['fp32_size_mb']:.2f} | {rn['fp16_size_mb']:.2f} | {rn['total_gmacs']:.3f} | {rn['total_gflops']:.3f} | None (from scratch) |",
        "",
        "### Architectural Differences Summary",
        "",
        "| Dimension | Simple CNN | AgriVision-CNN (Proposed) | ResNet18 Baseline |",
        "| :--- | :--- | :--- | :--- |",
        "| **Stem** | `3×3` Conv (32ch) + MaxPool | `3×3` Conv (32ch, s=2) | `7×7` Conv (64ch, s=2) + MaxPool |",
        "| **Multi-Scale Feature Block** | No | **Yes (Parallel `3×3` + `5×5` + `1×1` proj)** | No |",
        "| **Residual Connections** | No | **Yes (`64 → 128 → 256` with proj shortcuts)** | Yes (`64 → 128 → 256 → 512`) |",
        "| **Channel Attention (SE)** | No | **Yes (SE in every residual block, `r=8`)** | No |",
        "| **Parameter Efficiency vs. ResNet18** | "
        f"`{sc['total_params'] / rn['total_params'] * 100:.1f}%` of ResNet18 params | "
        f"**`{av['total_params'] / rn['total_params'] * 100:.1f}%`** of ResNet18 params (**{rn['total_params'] / av['total_params']:.2f}× smaller**) | `100.0%` (reference) |",
        "| **Accuracy / F1** | *Pending Phase 4 training* | *Pending Phase 4 training* | *Pending Phase 4 training* |",
        "",
        "> **Note on Empirical Performance:** No accuracy, precision, recall, or F1 claims are made in Phase 3. Controlled apples-to-apples training and evaluation on PlantVillage (`train`/`validation`/`test`) and external real-world `external_plantdoc` will be conducted in **Phase 4**.",
        "",
        "---",
        "",
        "## 3. AgriVision-CNN Layer-by-Layer Summary (`1 × 3 × 224 × 224`)",
        "",
        f"- **Input Tensor:** `{'×'.join(str(x) for x in av['input_shape'])}`",
        f"- **Output Logits Tensor:** `{'×'.join(str(x) for x in av['output_shape'])}`",
        f"- **Profiled Layers:** `{av['num_profiled_layers']}`",
        "",
        "| Layer Name | Layer Type | Input Shape | Output Shape | Parameters | MACs |",
        "| :--- | :--- | :--- | :--- | ---: | ---: |",
    ]

    for row in av["layers"]:
        lines.append(
            f"| `{row['name']}` | `{row['type']}` | `{row['input_shape']}` | "
            f"`{row['output_shape']}` | {row['params']:,} | {row['macs']:,} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Checkpoint & Reproducibility Design",
            "",
            "Reusable checkpoint utilities are implemented in `ml/src/models/checkpoint_utils.py` (`save_model_checkpoint` and `load_model_checkpoint`), storing:",
            "- `model_state_dict`",
            "- `model_config`",
            "- `class_names` and `class_mapping`",
            "- `random_seed`",
            "- `architecture_name`",
            "",
            "No trained weights have been created in `artifacts/` during Phase 3.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and compare Phase 3 CNN architectures.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to model_config.yaml.")
    args = parser.parse_args()

    config = load_model_config(args.config)
    project_root = Path(config["_project_root"])
    class_names, _ = resolve_verified_classes(config)

    h = int(config["input"]["image_height"])
    w = int(config["input"]["image_width"])
    c = int(config["input"]["in_channels"])
    input_size = (1, c, h, w)

    profiles: Dict[str, Dict[str, Any]] = {}
    print(f"=== Profiling Phase 3 Architectures (input={input_size}, num_classes={len(class_names)}) ===")
    for arch in SUPPORTED_ARCHITECTURES:
        model = create_model(architecture=arch, model_config=config)
        stats = profile_model(model, input_size=input_size)
        profiles[arch] = stats
        print(
            f"  - {arch:18s} | Params: {stats['total_params']:10,} | "
            f"FP32: {stats['fp32_size_mb']:6.2f} MB | "
            f"GMACs: {stats['total_gmacs']:6.3f} | GFLOPs: {stats['total_gflops']:6.3f} | "
            f"Out: {stats['output_shape']}"
        )

    diagram_path = project_root / "ml" / "reports" / "charts" / "agrivision_cnn_architecture.png"
    draw_agrivision_diagram(len(class_names), profiles["agrivision_cnn"], diagram_path)

    report_path = project_root / "ml" / "reports" / "model_architecture_report.md"
    write_architecture_report(
        config=config,
        class_names=class_names,
        profiles=profiles,
        report_path=report_path,
        diagram_rel_path="ml/reports/charts/agrivision_cnn_architecture.png",
    )

    print(f"\nSaved architecture diagram: {diagram_path}")
    print(f"Saved architecture report:  {report_path}")


if __name__ == "__main__":
    main()
