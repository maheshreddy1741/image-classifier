"""
predict.py
----------
Predict the class of one or more images using the trained model.

Usage
-----
    # Predict on 5 random CIFAR-10 test images (default demo)
    python predict.py

    # Predict on your own image file
    python predict.py --image path/to/image.jpg
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import load_model

from data_loader import load_data, CLASS_NAMES

MODEL_PATH  = os.path.join("saved_model", "best_model.h5")
IMG_SIZE    = (32, 32)           # CIFAR-10 native resolution


def load_and_preprocess_image(path: str) -> np.ndarray:
    """Load an image from disk, resize to 32×32, and normalise."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype("float32") / 255.0


def predict_single(model, img_array: np.ndarray) -> tuple[str, float]:
    """Run inference on a single normalised (32,32,3) array."""
    inp    = np.expand_dims(img_array, axis=0)           # (1, 32, 32, 3)
    probs  = model.predict(inp, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


def demo_on_test_images(model, n=5):
    """Predict on n random CIFAR-10 test images and save a figure."""
    _, _, x_test, y_test = load_data()
    indices = np.random.choice(len(x_test), n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for ax, idx in zip(axes, indices):
        img        = x_test[idx]
        true_label = CLASS_NAMES[int(y_test[idx])]
        pred_label, conf = predict_single(model, img)

        ax.imshow(img)
        color = "green" if pred_label == true_label else "red"
        ax.set_title(
            f"Pred: {pred_label}\n({conf*100:.1f}%)\nTrue: {true_label}",
            color=color, fontsize=9,
        )
        ax.axis("off")

    plt.suptitle("CIFAR-10 Predictions  (green = correct, red = wrong)",
                 fontsize=11)
    plt.tight_layout()
    out = "sample_predictions.png"
    plt.savefig(out, dpi=150)
    print(f"[predict] Prediction grid saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classifier — Predict")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image file to classify.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of random test images for the demo (default: 5).")
    args = parser.parse_args()

    # Load model ───────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[predict] ERROR: Model not found at '{MODEL_PATH}'.")
        print("[predict]        Run `python train.py` first.")
        sys.exit(1)

    print(f"[predict] Loading model from {MODEL_PATH} …")
    model = load_model(MODEL_PATH)

    if args.image:
        # Single custom image ──────────────────────────────────────────────────
        img             = load_and_preprocess_image(args.image)
        pred, conf      = predict_single(model, img)
        print(f"\n[predict] Image : {args.image}")
        print(f"[predict] Class : {pred}")
        print(f"[predict] Conf  : {conf*100:.2f}%")
    else:
        # Demo on random test images ───────────────────────────────────────────
        print(f"\n[predict] Running demo on {args.n} random CIFAR-10 test images …")
        demo_on_test_images(model, n=args.n)


if __name__ == "__main__":
    main()
