"""
dataset.py — Dataset class, transforms, and dataloaders.
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)

Assumes folder structure:
    data/agricultural_pests/
        ants/       *.jpg
        bees/       *.jpg
        beetles/    *.jpg
        ... (one subfolder per class, named exactly as config.class_names)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, InterpolationMode
from torchvision.utils import make_grid, save_image

from config import CFG, Config


# ---------------------------------------------------------------------------
# Pure-NumPy stratified split (avoids scipy/sklearn DLL issues on Windows)
# ---------------------------------------------------------------------------

def _stratified_split(
    paths: List,
    labels: List[int],
    test_size: float,
    seed: int,
) -> Tuple[List, List, List[int], List[int]]:
    """
    Stratified split without sklearn/scipy.
    Returns (train_paths, test_paths, train_labels, test_labels).
    """
    rng = np.random.default_rng(seed)
    paths_arr  = np.array(paths,  dtype=object)
    labels_arr = np.array(labels, dtype=int)

    train_idx, test_idx = [], []
    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())

    return (
        paths_arr[train_idx].tolist(),
        paths_arr[test_idx].tolist(),
        labels_arr[train_idx].tolist(),
        labels_arr[test_idx].tolist(),
    )

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PestDataset(Dataset):
    """
    Loads images from local folder structure.
    Performs stratified train/val/test split (70/15/15).

    Args:
        root_dir: Path to dataset root (contains one sub-folder per class).
        split:    One of 'train', 'val', 'test'.
        transform: torchvision transform pipeline.
        config:   Config dataclass instance.
        seed:     Random seed for reproducible splits.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        transform: transforms.Compose,
        config: Config,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.config = config

        # ---- Collect all (path, label) pairs ----------------------------
        all_samples: List[Tuple[Path, int]] = []
        for label_idx, class_name in enumerate(config.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s — skipping", class_dir)
                continue
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_samples.append((img_path, label_idx))

        if not all_samples:
            raise RuntimeError(
                f"No images found under '{root_dir}'. "
                "Check that data_root points to the correct folder."
            )

        paths, labels = zip(*all_samples)
        paths = list(paths)
        labels = list(labels)

        # ---- Stratified split (two-step, pure NumPy — no scipy needed) --
        # Step 1: 85% trainval / 15% test
        tv_paths, test_paths, tv_labels, test_labels = _stratified_split(
            paths, labels,
            test_size=config.test_split,
            seed=seed,
        )
        # Step 2: of trainval, 82.35% train / 17.65% val  → overall 70/15
        val_fraction_of_tv = config.val_split / (config.train_split + config.val_split)
        tr_paths, val_paths, tr_labels, val_labels = _stratified_split(
            tv_paths, tv_labels,
            test_size=val_fraction_of_tv,
            seed=seed + 1,
        )

        split_map = {
            "train": (tr_paths, tr_labels),
            "val":   (val_paths, val_labels),
            "test":  (test_paths, test_labels),
        }
        self.samples: List[Tuple[Path, int]] = list(
            zip(*split_map[split]) if split_map[split][0] else ([], [])
        )

        # ---- Log class distribution & compute weights -------------------
        class_counts = np.zeros(config.num_classes, dtype=int)
        for _, lbl in self.samples:
            class_counts[lbl] += 1

        logger.info("Split='%s'  total=%d", split, len(self.samples))
        for i, (name, cnt) in enumerate(zip(config.class_names, class_counts)):
            logger.info("  %-15s: %d", name, cnt)

        mean_count = class_counts.mean()
        deviation = np.abs(class_counts - mean_count) / (mean_count + 1e-8)
        self.use_weighted_sampler: bool = bool((deviation > 0.20).any())

        if self.use_weighted_sampler and split == "train":
            logger.info("Class imbalance detected (>20%% deviation) — "
                        "WeightedRandomSampler will be used for training.")
            self.class_weights = 1.0 / (class_counts + 1e-8)
            self.class_weights /= self.class_weights.sum()  # normalise
        else:
            self.class_weights = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns (image_tensor, label). Skips corrupt files gracefully."""
        max_tries = len(self.samples)
        for attempt in range(max_tries):
            try:
                img_path, label = self.samples[(idx + attempt) % len(self.samples)]
                image = Image.open(img_path).convert("RGB")
                tensor = self.transform(image)
                return tensor, label
            except Exception as exc:
                logger.warning("Corrupt/unreadable image '%s': %s — skipping",
                               img_path, exc)
        raise RuntimeError(f"Could not load any image starting at index {idx}")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str, config: Config) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline for a given split.

    Training: heavy augmentation as specified in the paper.
    Val/Test: deterministic centre-crop pipeline.
    """
    mean = config.imagenet_mean
    std = config.imagenet_std

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                config.image_size,
                scale=(config.crop_scale_min, config.crop_scale_max),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=config.flip_prob),
            transforms.RandomVerticalFlip(p=config.flip_prob),
            transforms.RandomRotation(
                config.rotation_degrees,
                interpolation=InterpolationMode.BILINEAR,
            ),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                config.resize_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(config: Config) -> Dict[str, DataLoader]:
    """
    Builds and returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.

    * Training loader uses WeightedRandomSampler if the dataset is imbalanced.
    * Saves a 4×4 grid of sample training images to results/sample_grid.png.
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, PestDataset] = {
        s: PestDataset(
            root_dir=config.data_root,
            split=s,
            transform=get_transforms(s, config),
            config=config,
            seed=config.random_seed,
        )
        for s in ("train", "val", "test")
    }

    train_ds = datasets["train"]

    # Weighted sampler for imbalanced training sets
    if train_ds.use_weighted_sampler and train_ds.class_weights is not None:
        sample_weights = torch.tensor(
            [train_ds.class_weights[lbl] for _, lbl in train_ds.samples],
            dtype=torch.float,
        )
        sampler: Optional[WeightedRandomSampler] = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),   # must equal len(dataset), NOT len(weights)
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=train_shuffle,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        ),
    }

    # ---- Save 4×4 sample grid -----------------------------------------
    try:
        images, _ = next(iter(loaders["train"]))
        grid_imgs = images[:16]
        # Denormalise for display
        mean_t = torch.tensor(config.imagenet_mean).view(1, 3, 1, 1)
        std_t = torch.tensor(config.imagenet_std).view(1, 3, 1, 1)
        grid_imgs = grid_imgs * std_t + mean_t
        grid_imgs = grid_imgs.clamp(0, 1)
        grid = make_grid(grid_imgs, nrow=4, padding=4)
        save_path = config.results_dir / "sample_grid.png"
        save_image(grid, str(save_path))
        logger.info("Sample grid saved to %s", save_path)
    except Exception as exc:
        logger.warning("Could not save sample grid: %s", exc)

    return loaders
