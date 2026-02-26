"""
model.py
--------
Defines the CNN architecture for CIFAR-10 classification.

Architecture
------------
3 × (Conv2D → BatchNorm → MaxPooling → Dropout) blocks
  → Flatten → Dense(256) → Dropout → Dense(10, softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(input_shape=(32, 32, 3), num_classes=10, learning_rate=1e-3):
    """
    Build and compile a CNN for image classification.

    Parameters
    ----------
    input_shape   : tuple  – (H, W, C)
    num_classes   : int    – number of output classes
    learning_rate : float  – Adam learning rate

    Returns
    -------
    model : compiled tf.keras.Sequential
    """
    model = models.Sequential(name="CIFAR10_CNN")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 2 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Block 3 ──────────────────────────────────────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.40))

    # ── Classifier head ──────────────────────────────────────────────────────
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


if __name__ == "__main__":
    m = build_model()
    print("\nModel built successfully.")
