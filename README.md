# 🖼️ Simple Image Classifier — CIFAR-10 with TensorFlow/Keras

A convolutional neural network (CNN) that classifies images into **10 categories** using the CIFAR-10 dataset.

---

## 📁 Project Structure

```
image_classifier/
├── data_loader.py       # Download & preprocess CIFAR-10
├── model.py             # CNN architecture
├── train.py             # Train the model
├── evaluate.py          # Evaluate on test set (accuracy, precision, recall, F1)
├── predict.py           # Predict on new or random images
├── requirements.txt     # Dependencies
└── saved_model/
    └── best_model.h5    # Generated after training
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model
```bash
python train.py
```
- Downloads CIFAR-10 automatically on first run
- Saves the best model to `saved_model/best_model.h5`
- Saves training curves to `training_history.png`

### 2. Evaluate on the test set
```bash
python evaluate.py
```
Prints **accuracy, precision, recall, F1-score** per class and saves `confusion_matrix.png`.

### 3. Predict on images
```bash
# Demo: predict on 5 random CIFAR-10 test images
python predict.py

# Predict on your own image
python predict.py --image path/to/your/image.jpg
```
Saves `sample_predictions.png` with results.

---

## 🏗️ CNN Architecture

| Block | Layers |
|-------|--------|
| Block 1 | Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 2 | Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25) |
| Block 3 | Conv2D(128) → BatchNorm → MaxPool → Dropout(0.40) |
| Head   | Flatten → Dense(256) → Dropout(0.50) → Dense(10, softmax) |

---

## 📊 CIFAR-10 Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

---

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | ~75–80% |
| Training Time | ~20–40 min (CPU) |

> **Tip:** Training on a GPU is highly recommended for faster results.

---

## 🔧 Hyperparameters (in `train.py`)

| Parameter | Default |
|-----------|---------|
| Epochs | 50 (w/ EarlyStopping) |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Augmentation | Flip, Rotation, Zoom |
