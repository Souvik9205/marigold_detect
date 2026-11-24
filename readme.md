# Marigold Plant Disease Detection using Hybrid ResNet50-SVM Model

## Project Overview

This project aims to develop an intelligent disease detection system for **Marigold (Tagetes)** plants using a **Hybrid Convolutional Neural Network (CNN) and Support Vector Machine (SVM)** architecture.

The proposed system utilises **ResNet50** as a deep feature extractor and **Support Vector Machine (SVM)** as the final classifier for disease prediction.

The initial phase focuses on **leaf disease classification**, while flower disease classification will be incorporated after preparing and annotating a suitable flower dataset.

---

# Dataset Details

## Dataset Source

The dataset was collected from multiple publicly available image datasets and manually cleaned before training.

Initially, two labelled datasets were merged to create a unified training dataset (**Dataset-F**).

Images were manually inspected to remove:

* Duplicate images
* Corrupted images
* Blurry images
* Incorrectly labelled samples
* Non-marigold images

---

## Final Leaf Dataset

| Class                  | Number of Images |
| ---------------------- | ---------------: |
| Alternaria Leaf Blight |          **338** |
| Healthy Leaf           |          **273** |
| Pest Damaged Leaf      |           **78** |
| **Total Images**       |          **689** |

### Class Distribution

* **Alternaria Leaf Blight:** 49.1%
* **Healthy Leaf:** 39.6%
* **Pest Damaged Leaf:** 11.3%

The dataset is moderately imbalanced, particularly for the Pest Damaged class. To reduce the effect of class imbalance during training, the SVM classifier uses `class_weight="balanced"`.

---

# Dataset Split

The dataset was divided using **stratified sampling** to preserve class distribution across all subsets.

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |        70% |
| Validation |        15% |
| Testing    |        15% |

Approximate image counts:

| Split      | Images |
| ---------- | -----: |
| Training   |    481 |
| Validation |    104 |
| Testing    |    104 |

Random seed:

```text
42
```

---

# Image Preprocessing

Each image undergoes the following preprocessing steps before feature extraction.

## Image Verification

Every image is checked using OpenCV to ensure that it can be successfully loaded.

Invalid or corrupted images are discarded automatically.

---

## Image Size

All images are resized to

```text
224 × 224 pixels
```

which is the standard input size required by ResNet50.

---

## Colour Format

Images are converted to

```text
RGB
```

before being passed to the neural network.

---

## Pixel Preprocessing

Images are preprocessed using TensorFlow's built-in ResNet50 preprocessing function:

```python
tensorflow.keras.applications.resnet50.preprocess_input()
```

This performs the same preprocessing used during ImageNet training.

---

# Feature Extraction

Instead of training a CNN from scratch, a pretrained **ResNet50** network is used.

Configuration:

| Parameter          | Value         |
| ------------------ | ------------- |
| Architecture       | ResNet50      |
| Pretrained Weights | ImageNet      |
| Include Top Layer  | False         |
| Input Size         | 224 × 224 × 3 |

The final fully connected classification layer is removed.

A **Global Average Pooling (GAP)** layer converts the convolutional feature maps into a fixed-length feature vector.

Each image is represented by a

```text
2048-dimensional feature vector
```

These feature vectors are then used for machine learning classification.

---

# Feature Scaling

The extracted feature vectors are standardised using

```text
StandardScaler
```

This ensures that every feature has zero mean and unit variance before training the Support Vector Machine.

---

# Classification Model

A Support Vector Machine (SVM) is used as the final classifier.

Current configuration:

| Parameter    | Value    |
| ------------ | -------- |
| Kernel       | RBF      |
| C            | 10       |
| Gamma        | scale    |
| Class Weight | balanced |

The use of a hybrid CNN-SVM architecture enables the model to leverage deep visual features extracted by ResNet50 while benefiting from the strong classification capabilities of SVM on relatively small datasets.

---

# Training Pipeline

```text
Raw Leaf Images
        │
        ▼
Image Verification
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Image Preprocessing
        │
        ▼
ResNet50 Feature Extraction
        │
        ▼
2048-D Feature Vector
        │
        ▼
StandardScaler
        │
        ▼
Support Vector Machine
        │
        ▼
Disease Prediction
```

---

# Model Performance (Baseline)

Initial evaluation was performed using the held-out test dataset.

## Overall Accuracy

```text
90.38%
```

### Classification Report

| Class        | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Alternaria   |      0.86 |   0.96 |     0.91 |
| Healthy      |      0.98 |   0.98 |     0.98 |
| Pest Damaged |      0.83 |   0.42 |     0.56 |

The lower recall for the Pest Damaged class is attributed to the limited number of available training samples and the similarity between pest-induced damage and fungal symptoms.

---

# Future Work

The following improvements are planned:

* Hyperparameter optimisation using GridSearchCV.
* Automatic labelling of additional leaf images using the trained baseline model followed by manual verification.
* Expansion of the training dataset with verified samples.
* Collection and annotation of flower disease images.
* Development of a flower disease classifier using the same hybrid CNN-SVM architecture.
* Comparative evaluation of different SVM kernels (Linear and RBF).
* Integration of PCA for feature dimensionality reduction (if it improves performance).

---

# Software and Libraries

| Component               | Technology         |
| ----------------------- | ------------------ |
| Programming Language    | Python 3.9         |
| Deep Learning Framework | TensorFlow / Keras |
| Machine Learning        | Scikit-learn       |
| Image Processing        | OpenCV             |
| Numerical Computing     | NumPy              |
| Data Handling           | Pandas             |
| Visualisation           | Matplotlib         |
| Model Persistence       | Joblib             |

---

# Current Project Status

* ✅ Dataset collection completed (leaf dataset)
* ✅ Dataset cleaning completed
* ✅ Stratified train/validation/test split implemented
* ✅ ResNet50 feature extraction implemented
* ✅ Hybrid ResNet50-SVM model implemented
* ✅ Baseline accuracy of **90.38%** achieved
* 🔄 Hyperparameter tuning in progress
* 🔄 Additional leaf dataset annotation in progress
* ⏳ Flower disease dataset preparation pending
* ⏳ Final hybrid model training pending

---

# Local Setup and Batch Prediction

Run the following commands from the repository root. The project documents Python 3.9 as its baseline environment; use a compatible TensorFlow installation for your platform.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell (instead of the preceding activation command)
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

`main.py` reports the installed TensorFlow, NumPy, OpenCV and scikit-learn versions and lists available GPUs. It is an environment check, not the training entry point.

## Running batch inference

1. Ensure `models/` contains `svm_classifier.pkl`, `scaler.pkl` and `label_encoder.pkl` from the same training run. Only load trusted Joblib artifacts.
2. Place input images directly in `data/dataset3/`. Supported extensions are `.jpg`, `.jpeg`, `.png` and `.bmp` (case-insensitive).
3. Run `python predict.py` from the repository root. ResNet50 ImageNet weights may require a network connection on first use.
4. Inspect `predictions.csv` for filenames, predicted labels and scores. Copies of input images are grouped by predicted class under `data/predicted/`.

The CSV `score` is the maximum SVM decision-function output, not a calibrated probability or percentage confidence. Manually review predicted labels before using them as new training annotations. The script overwrites `predictions.csv` on each run and may leave files from earlier runs in the output directories; preserve previous results separately if needed.
