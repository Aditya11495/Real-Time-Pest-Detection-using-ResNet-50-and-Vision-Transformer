"""
run_inference.py — Load trained checkpoint and run test evaluation and Grad-CAM.
"""
import csv
import sys
from pathlib import Path
import torch
import numpy as np

# Make sure the package directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from dataset import get_dataloaders
from evaluate import full_evaluation
from gradcam import visualize_gradcam
from model import TriPathFusionModel, print_model_summary


def main() -> None:
    config = CFG

    print("=" * 60)
    print("  Pest Detection — TriPath Fusion Model (Inference Only)")
    print("  IEEE TCE 2025 — ResNet-50 + ViT-B/16 + CustomCNN")
    print("=" * 60)

    # ---- 1. Device Setup ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 2. Build Model & Load Checkpoint -----------------------------------
    print("\nBuilding model ...")
    model = TriPathFusionModel(config).to(device)

    best_ckpt = config.checkpoint_dir / "best_model.pth"
    if not best_ckpt.exists():
        print(f"Error: Checkpoint not found at {best_ckpt}")
        return

    print(f"Loading best model weights from {best_ckpt} ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt.get("epoch", 1)
    print(f"Weights loaded successfully (saved at epoch {best_epoch}).")

    # ---- 3. Load Training History -------------------------------------------
    history_path = config.results_dir / "training_history.csv"
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr":         [],
    }

    if history_path.exists():
        print(f"Loading training history from {history_path} ...")
        with open(history_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in history.keys():
                    if k in row:
                        history[k].append(float(row[k]))
        print("History loaded successfully.")
    else:
        print(f"Warning: History file not found at {history_path}. Creating dummy history.")
        # Fallback dummy history if file is missing
        history = {
            "train_loss": [1.0], "val_loss": [ckpt.get("val_loss", 1.0)],
            "train_acc":  [90.0], "val_acc":  [90.0],
            "lr":         [config.learning_rate],
        }

    # ---- 4. Rebuild Test Loader ---------------------------------------------
    print("\nLoading datasets ...")
    loaders = get_dataloaders(config)
    test_loader = loaders["test"]

    # ---- 5. Run Full Evaluation ---------------------------------------------
    print("\n" + "=" * 60)
    print("  Running Full Evaluation on Test Set ...")
    print("=" * 60)

    summary = full_evaluation(
        model        = model,
        test_loader  = test_loader,
        history      = history,
        config       = config,
        device       = device,
        best_epoch   = best_epoch,
    )

    # ---- 6. Grad-CAM Visualisation ------------------------------------------
    print("\nGenerating Grad-CAM visualisations ...")
    try:
        visualize_gradcam(
            model        = model,
            test_loader  = test_loader,
            config       = config,
            device       = device,
            num_samples  = 8,
            save_path    = config.results_dir / "gradcam.png",
        )
        print(f"Grad-CAM saved to {config.results_dir / 'gradcam.png'}")
    except Exception as exc:
        print(f"  [Warning] Grad-CAM failed: {exc}")

    # ---- 7. Final Summary Box -----------------------------------------------
    test_acc  = summary.get("test_accuracy", 0) * 100
    macro_f1  = summary.get("macro_f1",      0) * 100
    max_ep    = config.max_epochs

    print(
        f"\n"
        f"╔══════════════════════════════════════╗\n"
        f"║     FINAL RESULTS SUMMARY            ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"║ Test Accuracy:    {test_acc:>6.2f}%             ║\n"
        f"║ Macro F1:         {macro_f1:>6.2f}%             ║\n"
        f"║ Best Epoch:       {best_epoch:>2d} / {max_ep:<2d}            ║\n"
        f"║ Results saved to: results/           ║\n"
        f"╚══════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    main()
