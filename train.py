"""
train.py — Full training loop with early stopping, AMP, and CosineAnnealingLR.
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)
"""
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import CFG, Config
from dataset import get_dataloaders
from model import TriPathFusionModel, print_model_summary

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from tqdm.notebook import tqdm as tqdm_nb
    from tqdm import tqdm
except ImportError:
    from tqdm import tqdm
    tqdm_nb = tqdm


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training if val_loss does not improve for `patience` epochs.
    Saves the best checkpoint automatically.

    Args:
        patience:  Epochs to wait without improvement (paper uses 5).
        min_delta: Minimum improvement threshold (1e-4).
        path:      Where to save the best model weights.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        path: Path = Path("checkpoints/best_model.pth"),
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = Path(path)
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False
        self.best_epoch: int = 0

    def __call__(
        self, val_loss: float, model: nn.Module, epoch: int
    ) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            self._save(model, epoch, val_loss)
        else:
            self.counter += 1
            logger.info("  EarlyStopping counter: %d / %d",
                        self.counter, self.patience)
            if self.counter >= self.patience:
                self.stop = True

    def _save(self, model: nn.Module, epoch: int, val_loss: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
            },
            self.path,
        )
        logger.info("  Checkpoint saved → %s  (val_loss=%.4f)", self.path, val_loss)


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    config: Config,
    epoch: int,
) -> Tuple[float, float]:
    """
    One full training epoch with AMP mixed precision.

    Per batch:
      1. images, labels → device (non_blocking=True)
      2. optimizer.zero_grad(set_to_none=True)
      3. with autocast(): logits = model(images); loss = criterion(logits, labels)
      4. scaler.scale(loss).backward()
      5. scaler.unscale_(optimizer)
      6. clip_grad_norm_(model.parameters(), config.gradient_clip)
      7. scaler.step(optimizer); scaler.update()

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0
    device_type  = device.type  # 'cuda' or 'cpu'

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=config.mixed_precision and device_type == 'cuda'):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validation pass — torch.no_grad(), model.eval().

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  [val]  ", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------

def main(config: Config = CFG) -> Tuple[TriPathFusionModel, Dict]:
    """
    Full training orchestration:

    1.  Set seeds for reproducibility.
    2.  Create results/ and checkpoints/ directories.
    3.  Load dataloaders.
    4.  Print class distribution table.
    5.  Build TriPathFusionModel, print summary.
    6.  Loss: CrossEntropyLoss(label_smoothing=0.1).
    7.  Optimizer: Adam(lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)).
    8.  Scheduler: CosineAnnealingLR(T_max=50, eta_min=1e-6).
    9.  Scaler: GradScaler.
    10. EarlyStopping(patience=5).
    11. History dict: {train_loss, val_loss, train_acc, val_acc, lr}.
    12. Epoch loop with formatted table output.
    13. Load best checkpoint weights.
    14. Save training history CSV.
    15. Plot training curves.

    Returns:
        model: Best-checkpoint model (on CPU for memory efficiency).
        history: Dictionary of per-epoch metrics.
    """
    # ---- 1. Seeds -------------------------------------------------------
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ---- 2. Directories -------------------------------------------------
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3. Dataloaders -------------------------------------------------
    logger.info("Loading datasets from '%s' ...", config.data_root)
    loaders = get_dataloaders(config)

    # ---- 4. Class distribution table ------------------------------------
    train_ds = loaders["train"].dataset
    print("\nClass distribution (training set):")
    print(f"  {'Class':<20} {'Count':>8}")
    print("  " + "-" * 30)
    for name, cnt in zip(
        config.class_names,
        np.bincount(
            [lbl for _, lbl in train_ds.samples],
            minlength=config.num_classes,
        ),
    ):
        print(f"  {name:<20} {cnt:>8}")
    print()

    # ---- 5. Model -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == 'cuda':
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    model = TriPathFusionModel(config).to(device)
    print_model_summary(model, config)

    # ---- 6-10. Loss / Optim / Scheduler / Scaler / ES ------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.scheduler_t_max,
        eta_min=config.scheduler_eta_min,
    )
    use_amp = config.mixed_precision and device.type == 'cuda'
    try:
        scaler = GradScaler(device=device.type, enabled=use_amp)   # PyTorch >= 2.3
    except TypeError:
        scaler = GradScaler(enabled=use_amp)                        # PyTorch < 2.3
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        path=config.checkpoint_dir / "best_model.pth",
    )

    # ---- 11. History ----------------------------------------------------
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    # ---- 12. Epoch loop -------------------------------------------------
    logger.info("Starting training for up to %d epochs ...", config.max_epochs)
    for epoch in range(1, config.max_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_epoch(
            model, loaders["train"], optimizer, criterion,
            scaler, device, config, epoch,
        )
        val_loss, val_acc = val_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        best_marker = " ← best" if val_loss < early_stopping.best_loss + early_stopping.min_delta else ""
        print(
            f"\nEpoch {epoch:02d}/{config.max_epochs:02d} | LR: {current_lr:.2e}\n"
            f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%\n"
            f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{best_marker}"
        )

        early_stopping(val_loss, model, epoch)
        if early_stopping.stop:
            logger.info(
                "Early stopping triggered after %d epochs without improvement.",
                config.early_stopping_patience,
            )
            break

    # ---- 13. Load best checkpoint ---------------------------------------
    best_ckpt = config.checkpoint_dir / "best_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded best model from epoch %d (val_loss=%.4f)",
            ckpt["epoch"], ckpt["val_loss"],
        )

    # ---- 15. Save history CSV ------------------------------------------
    csv_path = config.results_dir / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history.keys())
        writer.writeheader()
        for i in range(len(history["train_loss"])):
            writer.writerow({k: v[i] for k, v in history.items()})
    logger.info("Training history saved to %s", csv_path)

    return model, history


if __name__ == "__main__":
    main()
