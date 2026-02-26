"""
evaluate.py
-----------
Evaluate the trained model on the CIFAR-10 test set.

Outputs
-------
  • Classification report (accuracy, precision, recall, F1 per class)
  • Confusion matrix image  → confusion_matrix.png

Usage
-----
    python evaluate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tensorflow.keras.models import load_model

from data_loader import load_data, CLASS_NAMES

MODEL_PATH = os.path.join("saved_model", "best_model.h5")


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Save a colour-coded confusion matrix to disk."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — CIFAR-10 Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


def evaluate():
    # 1. Load data ─────────────────────────────────────────────────────────────
    _, _, x_test, y_test = load_data()

    # 2. Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run `python train.py` first."
        )
    print(f"[evaluate] Loading model from {MODEL_PATH} …")
    model = load_model(MODEL_PATH)

    # 3. Predictions ───────────────────────────────────────────────────────────
    print("[evaluate] Running predictions on test set …")
    y_pred_probs = model.predict(x_test, batch_size=128, verbose=1)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = y_test.flatten()

    # 4. Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  Overall Test Accuracy : {acc*100:.2f}%")
    print(f"{'='*55}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # 5. Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASS_NAMES)

    # 6. Per-class accuracy ────────────────────────────────────────────────────
    print("\n[evaluate] Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        mask     = (y_true == i)
        cls_acc  = accuracy_score(y_true[mask], y_pred[mask])
        print(f"  {cls:<12} : {cls_acc*100:.1f}%")


if __name__ == "__main__":
    evaluate()
