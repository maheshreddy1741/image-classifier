"""
data_loader.py
--------------
Loads and preprocesses the CIFAR-10 dataset.
Handles corrupted Keras cache automatically.
"""

import os
import shutil
import numpy as np

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Keras dataset cache locations
_KERAS_CACHE = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
_CIFAR_DIR   = os.path.join(_KERAS_CACHE, "cifar-10-batches-py")
_CIFAR_TAR   = os.path.join(_KERAS_CACHE, "cifar-10-python.tar.gz")


def _clear_cifar_cache():
    """Remove any partially-downloaded / corrupted CIFAR-10 cache files."""
    for path in [_CIFAR_DIR, _CIFAR_TAR]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    print("[data_loader] Cleared corrupted CIFAR-10 cache. Retrying …")


def load_data():
    """
    Load CIFAR-10 and normalise pixel values to [0, 1].
    Automatically retries once if the cached file is corrupted.

    Returns
    -------
    x_train : np.ndarray  shape (50000, 32, 32, 3)
    y_train : np.ndarray  shape (50000, 1)
    x_test  : np.ndarray  shape (10000, 32, 32, 3)
    y_test  : np.ndarray  shape (10000, 1)
    """
    from tensorflow.keras.datasets import cifar10

    print("[data_loader] Downloading / loading CIFAR-10 …")

    for attempt in range(2):
        try:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            break
        except (ValueError, Exception) as exc:
            if attempt == 0 and ("corrupted" in str(exc).lower() or "hash" in str(exc).lower()):
                _clear_cifar_cache()
                continue
            raise

    # Normalise to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    print(f"[data_loader] Training samples : {len(x_train)}")
    print(f"[data_loader] Test     samples : {len(x_test)}")
    print(f"[data_loader] Image shape      : {x_train.shape[1:]}")
    print(f"[data_loader] Classes          : {CLASS_NAMES}")
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print("Data loaded successfully.")
