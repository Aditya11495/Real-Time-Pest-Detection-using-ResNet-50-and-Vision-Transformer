"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)

Target layer: model.resnet.layer4[-1]
(Last bottleneck block, activation shape: [B, 2048, 7, 7])

Algorithm:
  1. Forward pass → capture activations A ∈ R^{2048×7×7}
  2. Backward pass w.r.t. target class score y^c
  3. Importance weights: α_k = mean over spatial dims of (∂y^c / ∂A_k)
  4. Heatmap: L^c = ReLU(Σ_k  α_k · A_k)
  5. Normalise L^c to [0, 1]
  6. Resize to 224×224 with bilinear interpolation
  7. Overlay on original image (jet colormap, alpha=0.4)

IMPORTANT: Do NOT run GradCAM inside autocast(). Call model.float() first.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, Config
from model import TriPathFusionModel


# ---------------------------------------------------------------------------
# Grad-CAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for TriPathFusionModel.

    Target layer: model.resnet.layer4[-1]

    Usage:
        cam = GradCAM(model)
        heatmap, pred_class, confidence = cam.generate(image_tensor)
    """

    def __init__(self, model: TriPathFusionModel) -> None:
        self.model       = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to the last ResNet-50 block."""
        target = self.model.resnet.layer4[-1]

        target.register_forward_hook(
            lambda m, inp, out: setattr(self, "activations", out.detach())
        )
        target.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, "gradients", grad_out[0].detach())
        )

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for a single image.

        Args:
            image_tensor: [1, 3, 224, 224] — normalised input tensor.
            target_class: int or None (uses argmax prediction if None).

        Returns:
            heatmap:    np.ndarray [224, 224] in [0, 1]
            pred_class: int — predicted class index
            confidence: float — softmax confidence of predicted class
        """
        # PITFALL #3: GradCAM must run in float32, not float16
        self.model.eval()
        self.model.float()
        image_tensor = image_tensor.float()
        image_tensor.requires_grad_(True)

        logits = self.model(image_tensor)
        probs  = torch.softmax(logits, dim=1)
        pred   = int(logits.argmax(dim=1).item())
        conf   = float(probs[0, pred].item())

        target = pred if target_class is None else target_class
        self.model.zero_grad()
        logits[0, target].backward()

        # α_k = global average pooling of gradients
        assert self.gradients is not None,   "Gradient hook did not fire."
        assert self.activations is not None, "Activation hook did not fire."

        alpha   = self.gradients.mean(dim=[2, 3], keepdim=True)   # [1, 2048, 1, 1]
        heatmap = (alpha * self.activations).sum(dim=1).squeeze()  # [7, 7]
        heatmap = torch.relu(heatmap)

        # Normalise to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)

        # Resize to 224×224
        heatmap_np = cv2.resize(
            heatmap.cpu().numpy(),
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )
        return heatmap_np, pred, conf


# ---------------------------------------------------------------------------
# Denormalisation helper
# ---------------------------------------------------------------------------

def denormalize(
    tensor: torch.Tensor,
    mean:   Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:    Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Converts a normalised [1, 3, H, W] or [3, H, W] tensor to a
    uint8 [H, W, 3] RGB numpy array suitable for matplotlib display.
    """
    t = tensor.clone().squeeze(0)   # [3, H, W]
    for c, m, s in zip(t, mean, std):
        c.mul_(s).add_(m)
    img = t.permute(1, 2, 0).clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Batch visualisation
# ---------------------------------------------------------------------------

def visualize_gradcam(
    model:       TriPathFusionModel,
    test_loader: DataLoader,
    config:      Config,
    device:      torch.device,
    num_samples: int = 8,
    save_path:   Path = Path("results/gradcam.png"),
) -> None:
    """
    Generates Grad-CAM visualisations for `num_samples` test images.

    Layout: 3 rows × num_samples columns
        Row 1: Original image (denormalised)
        Row 2: Heatmap (jet colormap)
        Row 3: Overlay (50% blend)

    Annotation per column:
        Top:    true class name (green=correct, red=wrong)
        Bottom: predicted class + confidence %

    Figure size: (num_samples*3, 10), DPI=150
    """
    cam = GradCAM(model.to(device))
    model.eval()

    # Collect num_samples images from loader
    sample_images:  list = []
    sample_labels:  list = []
    for imgs, lbls in test_loader:
        for img, lbl in zip(imgs, lbls):
            sample_images.append(img.unsqueeze(0))
            sample_labels.append(int(lbl.item()))
            if len(sample_images) >= num_samples:
                break
        if len(sample_images) >= num_samples:
            break

    fig, axes = plt.subplots(3, num_samples,
                              figsize=(num_samples * 3, 10))

    for col, (img_tensor, true_label) in enumerate(
        zip(sample_images, sample_labels)
    ):
        img_tensor = img_tensor.to(device)
        heatmap, pred_class, confidence = cam.generate(img_tensor)

        # Denormalised original
        orig_rgb = denormalize(img_tensor.cpu())

        # Colour heatmap
        heat_uint8  = (heatmap * 255).astype(np.uint8)
        heat_colour = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_colour = cv2.cvtColor(heat_colour, cv2.COLOR_BGR2RGB)

        # Overlay (alpha blend)
        overlay = (0.5 * orig_rgb.astype(float) +
                   0.5 * heat_colour.astype(float)).clip(0, 255).astype(np.uint8)

        axes[0, col].imshow(orig_rgb)
        axes[1, col].imshow(heat_colour)
        axes[2, col].imshow(overlay)

        correct       = (pred_class == true_label)
        title_color   = "green" if correct else "red"
        true_name     = config.class_names[true_label]
        pred_name     = config.class_names[pred_class]

        axes[0, col].set_title(true_name, color=title_color, fontsize=8,
                                fontweight="bold")
        axes[2, col].set_xlabel(
            f"{pred_name}\n{confidence*100:.1f}%", fontsize=7
        )

        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Heatmap",  fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("Overlay",  fontsize=10, fontweight="bold")

    plt.suptitle("Grad-CAM Visualisations (ResNet-50 layer4[-1])",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved to {save_path}")
