"""
main.py — Entry point. Run everything top to bottom.

Usage:
    python main.py

Steps:
    1. Load config
    2. Build dataloaders + print dataset statistics
    3. Build model and print summary
    4. Train (with early stopping, AMP, cosine LR)
    5. Load best checkpoint
    6. Full evaluation (8 outputs)
    7. Grad-CAM visualisation
    8. Print final summary box
"""
import sys
import torch
from pathlib import Path

# ---- Make sure the pest_detection package directory is on the path -------
sys.path.insert(0, str(Path(__file__).parent))

from config import CFG
from dataset import get_dataloaders
from evaluate import full_evaluation, save_training_curves
from gradcam import visualize_gradcam
from model import TriPathFusionModel, print_model_summary
from train import main as train_main


def main() -> None:
    """
    Full pipeline:
        1.  Load config
        2.  Build dataloaders
        3.  Print dataset statistics
        4.  Build model + print summary
        5.  Train with early stopping
        6.  Load best checkpoint
        7.  Full evaluation (all 8 outputs)
        8.  Grad-CAM visualisation
        9.  Print final summary
    """
    config = CFG

    # ---- 1-5: Training (train.py handles steps 1–15 internally) ---------
    print("=" * 60)
    print("  Pest Detection — TriPath Fusion Model")
    print("  IEEE TCE 2025 — ResNet-50 + ViT-B/16 + CustomCNN")
    print("=" * 60)

    model, history = train_main(config)

    # ---- 6. Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    # ---- 7. Full evaluation --------------------------------------------------
    print("\n" + "=" * 60)
    print("  Running Full Evaluation on Test Set ...")
    print("=" * 60)

    # Rebuild loaders (same splits as training — same seeds)
    loaders = get_dataloaders(config)

    # Determine best epoch from history
    import numpy as np
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    summary = full_evaluation(
        model        = model,
        test_loader  = loaders["test"],
        history      = history,
        config       = config,
        device       = device,
        best_epoch   = best_epoch,
    )

    # ---- 8. Grad-CAM visualisation ------------------------------------------
    print("\nGenerating Grad-CAM visualisations ...")
    try:
        visualize_gradcam(
            model        = model,
            test_loader  = loaders["test"],
            config       = config,
            device       = device,
            num_samples  = 8,
            save_path    = config.results_dir / "gradcam.png",
        )
    except Exception as exc:
        print(f"  [Warning] Grad-CAM failed: {exc}")

    # ---- 9. Final summary box -----------------------------------------------
    test_acc  = summary.get("test_accuracy", 0) * 100
    macro_f1  = summary.get("macro_f1",      0) * 100
    best_ep   = summary.get("best_epoch",    0)
    max_ep    = config.max_epochs

    print(
        f"\n"
        f"╔══════════════════════════════════════╗\n"
        f"║     FINAL RESULTS SUMMARY            ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"║ Test Accuracy:    {test_acc:>6.2f}%             ║\n"
        f"║ Macro F1:         {macro_f1:>6.2f}%             ║\n"
        f"║ Best Epoch:       {best_ep:>2d} / {max_ep:<2d}            ║\n"
        f"║ Results saved to: results/           ║\n"
        f"╚══════════════════════════════════════╝\n"
    )


if __name__ == "__main__":
    main()
