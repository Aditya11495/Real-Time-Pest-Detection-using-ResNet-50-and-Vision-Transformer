"""
model.py — TriPathFusionModel: ViT-B/16 + ResNet-50 + CustomCNN hybrid fusion.
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)

Architecture summary:
    Three parallel feature extractors process the SAME input image:
        Fv = ViT-B/16(x)   → [B, 768]   global context via attention
        Fr = ResNet-50(x)   → [B, 2048]  hierarchical spatial features
        Fc = CustomCNN(x)   → [B, 256]   local textures and edges
    Fusion (paper Eq. 1):
        Ffinal = σ(Wh · [Fv; Fr; Fc] + bh)
    Implemented as a linear fusion head: 3072 → 512 → 12 logits.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import CFG, Config


# ---------------------------------------------------------------------------
# 5a. CustomCNN
# ---------------------------------------------------------------------------

class CustomCNN(nn.Module):
    """
    Lightweight 3-block CNN that extracts fine-grained local textures
    and edge patterns. Complements ViT (global context) and ResNet
    (hierarchical features).

    Architecture (input: [B, 3, 224, 224]):
      Block 1: Conv2d(3→32,  3×3, pad=1) → BN → ReLU → MaxPool2d(2)
               Output: [B, 32, 112, 112]
      Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool2d(2)
               Output: [B, 64, 56, 56]
      Block 3: Conv2d(64→128,3×3, pad=1) → BN → ReLU → AdaptiveAvgPool2d(1)
               Output: [B, 128, 1, 1]
      Flatten: [B, 128]
      FC1: Linear(128→512) → ReLU → Dropout(0.5)
      FC2: Linear(512→256) → ReLU → Dropout(0.5)
      Output: [B, 256]
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.feature_dim: int = config.cnn_output_dim  # 256

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # input-size agnostic
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.cnn_dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.cnn_dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv2d, Xavier for Linear, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            features: [B, 256]
        """
        x = self.block1(x)       # [B, 32, 112, 112]
        x = self.block2(x)       # [B, 64, 56, 56]
        x = self.block3(x)       # [B, 128, 1, 1]
        x = self.classifier(x)   # [B, 256]
        return x


# ---------------------------------------------------------------------------
# 5b. ViT-B/16 pathway
# ---------------------------------------------------------------------------

def get_vit(config: Config) -> Tuple[nn.Module, int]:
    """
    Loads pretrained ViT-B/16. Strips the classification head.

    ViT splits 224×224 image into 196 patches (14×14 grid, each 16×16 px).
    Each patch is linearly projected to a 768-dim embedding.
    Positional encodings added. 12 transformer encoder layers with
    12 attention heads each. [CLS] token output = 768-dim feature vector.

    IMPORTANT: uses BICUBIC interpolation — ViT was pretrained with it.

    Returns:
        model: ViT-B/16 with head replaced by Identity.
        feature_dim: 768.
    """
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # PITFALL #1: path is model.heads.head, NOT model.head
    model.heads.head = nn.Identity()  # strip classifier → [B, 768]
    return model, 768


# ---------------------------------------------------------------------------
# 5c. ResNet-50 pathway
# ---------------------------------------------------------------------------

def get_resnet(config: Config) -> Tuple[nn.Module, int]:
    """
    Loads pretrained ResNet-50. Strips the final FC layer.

    ResNet-50 architecture:
        conv1 → bn1 → relu → maxpool
        layer1 (3 bottleneck blocks)
        layer2 (4 bottleneck blocks)
        layer3 (6 bottleneck blocks)
        layer4 (3 bottleneck blocks)  ← Grad-CAM target
        avgpool → [B, 2048]  (after fc = Identity)

    Residual block: Bl(x) = ReLU(x + Fl(x))
    Total: 50 layers, output feature dim = 2048

    Returns:
        model: ResNet-50 with fc replaced by Identity.
        feature_dim: 2048.
    """
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()  # output is now [B, 2048]

    # PITFALL #2: assert shape is correct
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (1, 2048), \
            f"ResNet-50 output should be [B, 2048], got {out.shape}"

    return model, 2048


# ---------------------------------------------------------------------------
# 5d. TriPathFusionModel — THE MAIN MODEL
# ---------------------------------------------------------------------------

class TriPathFusionModel(nn.Module):
    """
    Triple-pathway hybrid fusion model.

    Three parallel feature extractors process the SAME input image:
      - Fv = ViT-B/16(x)     → [B, 768]   global context via attention
      - Fr = ResNet-50(x)     → [B, 2048]  hierarchical spatial features
      - Fc = CustomCNN(x)     → [B, 256]   local textures and edges

    Fusion formula (paper Eq. 1):
        Ffinal = σ(Wh · [Fv; Fr; Fc] + bh)
        where σ = softmax, [;] = concatenation

    Fusion head:
        concat([Fv, Fr, Fc])  → [B, 3072]
        Linear(3072 → 512)    → [B, 512]
        BatchNorm1d(512)
        ReLU
        Dropout(0.3)
        Linear(512 → 12)      → [B, 12]  (logits)

    Total parameters: ~98M
        ViT-B/16:    86M
        ResNet-50:   25M
        CustomCNN:   ~0.5M
        Fusion head: ~1.6M

    Why three paths?
        ViT alone misses fine-grained local detail (patch-level only).
        ResNet alone lacks long-range global context.
        CustomCNN adds low-level texture/edge info both miss.
        Together: complementary features → superior accuracy.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.vit,    vit_dim    = get_vit(config)      # [B, 768]
        self.resnet, resnet_dim = get_resnet(config)   # [B, 2048]
        self.cnn                = CustomCNN(config)    # [B, 256]
        cnn_dim                 = self.cnn.feature_dim

        total_dim = vit_dim + resnet_dim + cnn_dim     # 3072

        self.fusion_head = nn.Sequential(
            nn.Linear(total_dim, config.fusion_hidden_dim),   # 3072 → 512
            nn.BatchNorm1d(config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),  # 512 → 12
        )
        self._init_fusion_head()

    def _init_fusion_head(self) -> None:
        """Xavier initialisation for fusion head linear layers."""
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] — normalised input image batch
        Returns:
            logits: [B, 12] — raw class scores (apply softmax for probs)
        """
        fv = self.vit(x)       # [B, 768]
        fr = self.resnet(x)    # [B, 2048]
        fc = self.cnn(x)       # [B, 256]

        combined = torch.cat([fv, fr, fc], dim=1)  # [B, 3072]
        logits   = self.fusion_head(combined)       # [B, 12]
        return logits

    def get_feature_vectors(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns intermediate features — useful for t-SNE and ablation."""
        with torch.no_grad():
            return {
                "vit":    self.vit(x),
                "resnet": self.resnet(x),
                "cnn":    self.cnn(x),
            }


# ---------------------------------------------------------------------------
# Model summary utility
# ---------------------------------------------------------------------------

def print_model_summary(model: nn.Module, config: Config) -> None:
    """Prints layer shapes, param counts, and model size using torchinfo (optional)."""
    try:
        from torchinfo import summary
        summary(
            model,
            input_size=(1, 3, config.image_size, config.image_size),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=3,
        )
    except ImportError:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} total | {trainable:,} trainable")
        print("(Install torchinfo for a detailed layer-by-layer summary)")


# ---------------------------------------------------------------------------
# 5e. Ablation models — baselines for comparison table
# ---------------------------------------------------------------------------

class ViTOnlyModel(nn.Module):
    """ViT-B/16 → Linear(768, 12). Baseline for ablation study."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        vit_model, vit_dim = get_vit(config)
        self.vit = vit_model
        self.head = nn.Linear(vit_dim, config.num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


class ResNetOnlyModel(nn.Module):
    """ResNet-50 → Linear(2048, 12). Baseline for ablation study."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        resnet_model, resnet_dim = get_resnet(config)
        self.resnet = resnet_model
        self.head = nn.Linear(resnet_dim, config.num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.resnet(x))


class CNNOnlyModel(nn.Module):
    """CustomCNN → Linear(256, 12). Baseline for ablation study."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.cnn = CustomCNN(config)
        self.head = nn.Linear(self.cnn.feature_dim, config.num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))
