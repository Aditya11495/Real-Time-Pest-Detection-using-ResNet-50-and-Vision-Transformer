"""
config.py — All hyperparameters in one place.
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ Paths
    # Paths — data_root points to the folder containing the 12 class sub-folders
    data_root: Path = Path("../data")          # relative to pest_detection/
    checkpoint_dir: Path = Path("checkpoints")
    results_dir: Path = Path("results")

    # --------------------------------------------------------------- Dataset
    # Folder names on disk (as found in data/)
    class_names: List[str] = field(default_factory=lambda: [
        "ants", "bees", "beetle", "catterpillar", "earthworms",
        "earwig", "grasshopper", "moth", "slug", "snail",
        "wasp", "weevil"
    ])
    num_classes: int = 12
    image_size: int = 224
    resize_size: int = 256
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # ------------------------------------------------------------ Augmentation
    crop_scale_min: float = 0.6
    crop_scale_max: float = 1.0
    flip_prob: float = 0.5
    rotation_degrees: int = 30
    imagenet_mean: Tuple = (0.485, 0.456, 0.406)
    imagenet_std: Tuple = (0.229, 0.224, 0.225)

    # ----------------------------------------------------------- Model dims
    vit_output_dim: int = 768
    resnet_output_dim: int = 2048
    cnn_output_dim: int = 256
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3
    cnn_dropout: float = 0.5

    # ------------------------------------------------------------ Training
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    max_epochs: int = 8
    early_stopping_patience: int = 5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    scheduler_t_max: int = 8
    scheduler_eta_min: float = 1e-6


# Singleton instance imported by all other modules
CFG = Config()
