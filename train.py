"""
train.py
--------
Train the CNN on CIFAR-10 with:
  • tf.data pipeline (random flip, crop, brightness augmentation)
  • ModelCheckpoint  → saves best weights
  • EarlyStopping    → halts when val_loss stops improving
  • ReduceLROnPlateau→ lowers LR on plateau

Usage
-----
    python train.py
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

from data_loader import load_data
from model import build_model

# ── Hyper-parameters ─────────────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.1           # 10 % of training data used for validation
SAVE_DIR      = "saved_model"
MODEL_PATH    = os.path.join(SAVE_DIR, "best_model.h5")

os.makedirs(SAVE_DIR, exist_ok=True)


# ── tf.data augmentation helpers ─────────────────────────────────────────────

def augment(image, label):
    """Apply random augmentations to a single (image, label) pair."""
    # Pad 4 px each side then random-crop back to 32×32
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_datasets(x_train, y_train, batch_size, val_split):
    """Build train and validation tf.data.Dataset objects."""
    n_total = len(x_train)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    # Convert to float32 tensors
    x = tf.constant(x_train, dtype=tf.float32)
    y = tf.constant(y_train.flatten(), dtype=tf.int32)

    full_ds = tf.data.Dataset.from_tensor_slices((x, y))
    full_ds = full_ds.shuffle(n_total, reshuffle_each_iteration=False, seed=42)

    train_ds = (
        full_ds.take(n_train)
        .cache()
        .shuffle(n_train, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        full_ds.skip(n_train)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, n_train, n_val


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, save_path="training_history.png"):
    """Plot accuracy and loss curves and save to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[train] Training curves saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print(f"[train] TensorFlow {tf.__version__}")
    print(f"[train] GPUs available: {tf.config.list_physical_devices('GPU')}")

    # 1. Load data ─────────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_data()

    # 2. Build tf.data pipelines ───────────────────────────────────────────────
    train_ds, val_ds, n_train, n_val = make_datasets(
        x_train, y_train, BATCH_SIZE, VAL_SPLIT
    )
    print(f"[train] Train samples: {n_train}  |  Val samples: {n_val}")

    # 3. Build model ───────────────────────────────────────────────────────────
    model = build_model(learning_rate=LEARNING_RATE)

    # 4. Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Train ─────────────────────────────────────────────────────────────────
    print(f"\n[train] Starting training  (epochs={EPOCHS}, batch={BATCH_SIZE})\n")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # 6. Plot curves ───────────────────────────────────────────────────────────
    plot_history(history)

    # 7. Quick test-set peek ───────────────────────────────────────────────────
    x_test_t = tf.constant(x_test, dtype=tf.float32)
    y_test_t = tf.constant(y_test.flatten(), dtype=tf.int32)
    test_ds  = tf.data.Dataset.from_tensor_slices((x_test_t, y_test_t)).batch(128)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n[train] Test accuracy : {test_acc*100:.2f}%")
    print(f"[train] Test loss     : {test_loss:.4f}")
    print(f"[train] Best model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()

