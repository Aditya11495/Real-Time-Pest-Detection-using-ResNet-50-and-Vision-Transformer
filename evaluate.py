"""
evaluate.py — Comprehensive evaluation and visualisation after training.
Paper: "Real-Time Pest Detection Using ResNet-50 and Vision Transformer" (IEEE TCE 2025)

Outputs (all saved to results/):
  1.  classification_report.txt   — per-class precision/recall/F1
  2.  confusion_matrix.png        — 12×12 heatmap
  3.  training_curves.png         — loss + accuracy curves
  4.  per_class_f1.png            — horizontal bar chart
  5.  top_misclassifications.png  — 12 most confident errors
  6.  roc_curves.png              — one-vs-rest ROC for all classes
  7.  tsne_features.png           — t-SNE of fusion features
  8.  metrics_summary.json        — JSON summary of key metrics
  9.  model_comparison.png        — ablation bar chart (optional)
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless/Colab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from config import CFG, Config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pure-NumPy metric helpers (no sklearn / no scipy needed)
# ---------------------------------------------------------------------------

def _confusion_matrix_np(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute confusion matrix without sklearn."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def _classification_report_np(
    labels: np.ndarray,
    preds:  np.ndarray,
    class_names: List[str],
    digits: int = 4,
) -> Tuple[str, Dict]:
    """
    Compute per-class precision, recall, F1 and support without sklearn.
    Returns (report_string, report_dict).
    """
    n = len(class_names)
    report_dict: Dict = {}
    rows = []

    for i, name in enumerate(class_names):
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((preds != i) & (labels == i)).sum())
        support = int((labels == i).sum())

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)

        report_dict[name] = {"precision": prec, "recall": rec,
                             "f1-score": f1, "support": support}
        rows.append((name, prec, rec, f1, support))

    # Macro averages
    mac_p = np.mean([report_dict[n]["precision"] for n in class_names])
    mac_r = np.mean([report_dict[n]["recall"]    for n in class_names])
    mac_f = np.mean([report_dict[n]["f1-score"]  for n in class_names])
    tot_s = int(len(labels))
    report_dict["macro avg"]    = {"precision": mac_p, "recall": mac_r,
                                    "f1-score": mac_f,  "support": tot_s}
    # Weighted averages
    supports = np.array([report_dict[n]["support"] for n in class_names], dtype=float)
    w = supports / (supports.sum() + 1e-12)
    wgt_p = sum(w[i] * report_dict[n]["precision"] for i, n in enumerate(class_names))
    wgt_r = sum(w[i] * report_dict[n]["recall"]    for i, n in enumerate(class_names))
    wgt_f = sum(w[i] * report_dict[n]["f1-score"]  for i, n in enumerate(class_names))
    report_dict["weighted avg"] = {"precision": wgt_p, "recall": wgt_r,
                                    "f1-score": wgt_f,  "support": tot_s}

    acc = float((labels == preds).mean())

    # Format string
    w_name = max(len(n) for n in class_names) + 2
    header = f"{'':>{w_name}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}"
    sep    = "-" * len(header)
    lines  = [header, ""]
    for name, prec, rec, f1, sup in rows:
        lines.append(f"{name:>{w_name}} {prec:{10}.{digits}f} {rec:{10}.{digits}f} "
                     f"{f1:{10}.{digits}f} {sup:>9}")
    lines += [
        "",
        f"{'accuracy':>{w_name}} {'':>10} {'':>10} {acc:{10}.{digits}f} {tot_s:>9}",
        f"{'macro avg':>{w_name}} {mac_p:{10}.{digits}f} {mac_r:{10}.{digits}f} "
        f"{mac_f:{10}.{digits}f} {tot_s:>9}",
        f"{'weighted avg':>{w_name}} {wgt_p:{10}.{digits}f} {wgt_r:{10}.{digits}f} "
        f"{wgt_f:{10}.{digits}f} {tot_s:>9}",
        "",
    ]
    return "\n".join(lines), report_dict


def _roc_curve_np(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Binary ROC curve and AUC via trapezoidal rule. No sklearn needed."""
    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    pos = y_true.sum()
    neg = len(y_true) - pos
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / (pos + 1e-12))
        fpr_list.append(fp / (neg + 1e-12))
    tpr_list.append(1.0); fpr_list.append(1.0)
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _tsne_numpy(features: np.ndarray, n_iter: int = 500,
                perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """
    Minimal t-SNE using sklearn if available, otherwise a placeholder PCA-like
    projection so the plot still renders without scipy.
    """
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=seed).fit_transform(features)
    except Exception:
        pass
    # Fallback: PCA via SVD (always works, no sklearn/scipy needed)
    logger.warning("sklearn TSNE unavailable — using PCA projection instead.")
    X = features - features.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on `loader` and collect ground-truth labels,
    predicted class indices, softmax probabilities, and raw images.

    Returns:
        all_labels:  [N]        integer class indices
        all_preds:   [N]        predicted class indices
        all_probs:   [N, C]     softmax probabilities
        all_images:  [N, 3, H, W]  normalised image tensors (CPU)
    """
    model.eval()
    all_labels, all_preds, all_probs, all_images = [], [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collecting predictions", leave=False):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_images.append(images.cpu())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        torch.cat(all_images).numpy(),
    )


def _denormalize(
    img: np.ndarray,
    mean: Tuple = (0.485, 0.456, 0.406),
    std: Tuple  = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Converts a [C, H, W] normalised float32 array to uint8 [H, W, C] RGB."""
    img = img.copy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Output 1 — Classification report
# ---------------------------------------------------------------------------

def _save_classification_report(
    labels: np.ndarray,
    preds:  np.ndarray,
    config: Config,
) -> Dict:
    """Prints and saves per-class classification report."""
    report_str, report_dict = _classification_report_np(
        labels, preds,
        class_names=config.class_names,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report_str)

    path = config.results_dir / "classification_report.txt"
    path.write_text(report_str)
    logger.info("Classification report saved to %s", path)
    return report_dict


# ---------------------------------------------------------------------------
# Output 2 — Confusion matrix
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    labels: np.ndarray,
    preds:  np.ndarray,
    config: Config,
) -> None:
    """Saves a 12×12 annotated confusion matrix heatmap."""
    cm = _confusion_matrix_np(labels, preds, config.num_classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{cm_norm[i, j]:.1f}%"

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=annot, fmt="",
        cmap="Blues",
        xticklabels=config.class_names,
        yticklabels=config.class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    path = config.results_dir / "confusion_matrix.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)


# ---------------------------------------------------------------------------
# Output 3 — Training curves
# ---------------------------------------------------------------------------

def save_training_curves(history: Dict[str, List[float]], config: Config) -> None:
    """
    Plots and saves two-subplot training curves:
        Left:  Loss (train=solid blue, val=dashed orange)
        Right: Accuracy (train=solid blue, val=dashed orange)
    Marks the best epoch with a vertical dashed line.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch_idx = int(np.argmin(history["val_loss"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-",  label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   "orange", linestyle="--",
                 label="Val Loss")
    axes[0].axvline(best_epoch_idx + 1, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch {best_epoch_idx + 1}")
    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-",  label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "orange", linestyle="--",
                 label="Val Acc")
    axes[1].axvline(best_epoch_idx + 1, color="gray", linestyle="--", alpha=0.7,
                    label=f"Best epoch {best_epoch_idx + 1}")
    axes[1].set_title("Accuracy (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = config.results_dir / "training_curves.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", path)


# ---------------------------------------------------------------------------
# Output 4 — Per-class F1 bar chart
# ---------------------------------------------------------------------------

def _save_per_class_f1(report_dict: Dict, config: Config) -> None:
    """Horizontal bar chart of F1 per class, sorted worst→best."""
    f1_scores = [report_dict[c]["f1-score"] for c in config.class_names]
    sorted_pairs = sorted(zip(f1_scores, config.class_names))
    f1_sorted, names_sorted = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names_sorted)))
    bars = ax.barh(names_sorted, f1_sorted, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1 Score (sorted worst→best)", fontsize=14,
                 fontweight="bold")
    for bar, val in zip(bars, f1_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = config.results_dir / "per_class_f1.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class F1 chart saved to %s", path)


# ---------------------------------------------------------------------------
# Output 5 — Top misclassifications
# ---------------------------------------------------------------------------

def _save_top_misclassifications(
    labels:  np.ndarray,
    preds:   np.ndarray,
    probs:   np.ndarray,
    images:  np.ndarray,
    config:  Config,
    n:       int = 12,
) -> None:
    """Finds the n most confidently wrong predictions and plots a 3×4 grid."""
    wrong_mask = (preds != labels)
    wrong_idx  = np.where(wrong_mask)[0]
    if len(wrong_idx) == 0:
        logger.info("No misclassifications found on test set — skipping plot.")
        return

    # Sort by confidence of the wrong prediction (descending)
    confidences = probs[wrong_idx, preds[wrong_idx]]
    sorted_order = np.argsort(confidences)[::-1][:n]
    top_idx = wrong_idx[sorted_order]

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    axes = axes.flatten()

    for ax, idx in zip(axes, top_idx):
        img_rgb = _denormalize(images[idx])
        ax.imshow(img_rgb)
        true_name = config.class_names[labels[idx]]
        pred_name = config.class_names[preds[idx]]
        conf_pct  = confidences[list(wrong_idx).index(idx)
                                if idx in wrong_idx else 0] * 100
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}\n({probs[idx, preds[idx]]*100:.1f}%)",
            fontsize=7,
            color="red",
        )
        ax.axis("off")

    # Hide leftover axes
    for ax in axes[len(top_idx):]:
        ax.axis("off")

    plt.suptitle("Top Misclassifications (most confident errors)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = config.results_dir / "top_misclassifications.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Top misclassifications saved to %s", path)


# ---------------------------------------------------------------------------
# Output 6 — ROC curves
# ---------------------------------------------------------------------------

def _save_roc_curves(
    labels: np.ndarray,
    probs:  np.ndarray,
    config: Config,
) -> None:
    """One-vs-rest ROC curves for all 12 classes plus macro-average AUC."""
    n_classes = config.num_classes
    # Binarise labels one-vs-rest (pure numpy)
    lb_labels = np.eye(n_classes, dtype=int)[labels]  # [N, C]

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    aucs = []
    for i, (name, color) in enumerate(zip(config.class_names, colors)):
        try:
            fpr, tpr, auc = _roc_curve_np(lb_labels[:, i], probs[:, i])
            aucs.append(auc)
            ax.plot(fpr, tpr, color=color, lw=1.5,
                    label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass

    if aucs:
        macro_auc = np.mean(aucs)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_title(f"ROC Curves (Macro AUC={macro_auc:.3f})", fontsize=14,
                     fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = config.results_dir / "roc_curves.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved to %s", path)


# ---------------------------------------------------------------------------
# Output 7 — t-SNE
# ---------------------------------------------------------------------------

def _save_tsne(
    model:   nn.Module,
    loader:  DataLoader,
    config:  Config,
    device:  torch.device,
    n_samples: int = 300,
) -> None:
    """
    Extracts 256-dim CNN features from up to n_samples test images,
    runs t-SNE, and saves a scatter plot coloured by true class.
    """
    model.eval()
    features_list, label_list = [], []
    collected = 0

    with torch.no_grad():
        for images, labels in loader:
            if collected >= n_samples:
                break
            images = images.to(device)
            # Use CNN branch feature (256-dim) for t-SNE
            feats = model.cnn(images)
            features_list.append(feats.cpu().numpy())
            label_list.append(labels.numpy())
            collected += images.size(0)

    features = np.concatenate(features_list)[:n_samples]
    lab_arr  = np.concatenate(label_list)[:n_samples]

    logger.info("Running t-SNE on %d samples ...", len(features))
    embedded = _tsne_numpy(features, n_iter=1000, perplexity=30.0,
                           seed=config.random_seed)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, config.num_classes))
    for i, (name, color) in enumerate(zip(config.class_names, colors)):
        mask = lab_arr == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[color], label=name, alpha=0.7, s=20, edgecolors="none")
    ax.set_title("t-SNE of CNN Fusion Features (test set)", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    path = config.results_dir / "tsne_features.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("t-SNE plot saved to %s", path)


# ---------------------------------------------------------------------------
# Output 8 — metrics_summary.json
# ---------------------------------------------------------------------------

def _save_metrics_json(
    report_dict: Dict,
    labels:      np.ndarray,
    preds:       np.ndarray,
    best_epoch:  int,
    config:      Config,
) -> Dict:
    """Saves a compact JSON summary of key metrics."""
    test_acc   = float((labels == preds).mean())
    macro_f1   = float(report_dict["macro avg"]["f1-score"])
    weighted_f1 = float(report_dict["weighted avg"]["f1-score"])

    per_class = {
        name: {
            "precision": round(report_dict[name]["precision"], 4),
            "recall":    round(report_dict[name]["recall"],    4),
            "f1":        round(report_dict[name]["f1-score"],  4),
        }
        for name in config.class_names
        if name in report_dict
    }

    summary = {
        "test_accuracy":  round(test_acc,    4),
        "macro_f1":       round(macro_f1,    4),
        "weighted_f1":    round(weighted_f1, 4),
        "best_epoch":     best_epoch,
        "per_class":      per_class,
    }

    path = config.results_dir / "metrics_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    logger.info("Metrics summary saved to %s", path)
    return summary


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def full_evaluation(
    model:      nn.Module,
    test_loader: DataLoader,
    history:    Dict[str, List[float]],
    config:     Config,
    device:     torch.device,
    best_epoch: int = 0,
) -> Dict:
    """
    Runs all 8 evaluation outputs.

    Args:
        model:       Trained model (best checkpoint already loaded).
        test_loader: DataLoader for the test split.
        history:     Training history dict from train.py.
        config:      Config instance.
        device:      torch device.
        best_epoch:  Epoch index of best checkpoint (for JSON summary).

    Returns:
        summary: Dictionary with key metrics.
    """
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Collect predictions
    logger.info("Running inference on test set ...")
    labels, preds, probs, images = _collect_predictions(model, test_loader, device)

    # 1. Classification report
    report_dict = _save_classification_report(labels, preds, config)

    # 2. Confusion matrix
    _save_confusion_matrix(labels, preds, config)

    # 3. Training curves
    if history:
        save_training_curves(history, config)

    # 4. Per-class F1 bar chart
    _save_per_class_f1(report_dict, config)

    # 5. Top misclassifications
    _save_top_misclassifications(labels, preds, probs, images, config)

    # 6. ROC curves
    _save_roc_curves(labels, probs, config)

    # 7. t-SNE
    _save_tsne(model, test_loader, config, device)

    # 8. Metrics JSON
    summary = _save_metrics_json(report_dict, labels, preds, best_epoch, config)

    return summary


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def compare_models(
    loaders: Dict,
    config:  Config,
    device:  torch.device,
    epochs:  int = 10,
) -> None:
    """
    Trains ViTOnly, ResNetOnly, CNNOnly, and TriPathFusion models with the
    same config and saves a comparison bar chart to results/model_comparison.png.

    NOTE: For a fair ablation study this re-trains each model for `epochs`
    epochs; set epochs=config.max_epochs for a full run.
    """
    from model import (
        CNNOnlyModel,
        ResNetOnlyModel,
        TriPathFusionModel,
        ViTOnlyModel,
    )

    models_to_compare = {
        "CustomCNN only":  CNNOnlyModel(config),
        "ResNet-50 only":  ResNetOnlyModel(config),
        "ViT-B/16 only":   ViTOnlyModel(config),
        "Ours (TriPath)":  TriPathFusionModel(config),
    }

    val_accs: Dict[str, float] = {}
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    for name, mdl in models_to_compare.items():
        logger.info("Training ablation model: %s", name)
        mdl = mdl.to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
        scaler = GradScaler(enabled=config.mixed_precision)

        best_val_acc = 0.0
        for ep in range(1, epochs + 1):
            mdl.train()
            for imgs, lbls in loaders["train"]:
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=config.mixed_precision):
                    loss = criterion(mdl(imgs), lbls)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()

            # Val accuracy
            mdl.eval()
            correct = total = 0
            with torch.no_grad():
                for imgs, lbls in loaders["val"]:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    preds = mdl(imgs).argmax(dim=1)
                    correct += (preds == lbls).sum().item()
                    total   += lbls.size(0)
            val_acc = 100.0 * correct / total
            best_val_acc = max(best_val_acc, val_acc)
            logger.info("  [%s] epoch %d/%d  val_acc=%.2f%%",
                        name, ep, epochs, val_acc)

        val_accs[name] = best_val_acc
        mdl.cpu()
        torch.cuda.empty_cache()

    # Bar chart
    names   = list(val_accs.keys())
    accs    = [val_accs[n] for n in names]
    colors  = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.7)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Ablation Study — Model Comparison", fontsize=14,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = config.results_dir / "model_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", path)
