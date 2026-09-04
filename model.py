from pathlib import Path
import joblib
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)

# ----------------------------------------------------
# Load ResNet50
# ----------------------------------------------------

print("Loading ResNet50...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

feature_extractor = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output),
)

print("ResNet50 Loaded.\n")

# ----------------------------------------------------
# Feature Extraction
# ----------------------------------------------------


def extract_features(dataset_path):

    features = []
    labels = []

    classes = sorted(
        [folder.name for folder in dataset_path.iterdir() if folder.is_dir()]
    )

    for cls in classes:

        folder = dataset_path / cls

        print(f"Processing {cls}")

        for img_path in folder.iterdir():

            img = image.load_img(
                img_path,
                target_size=IMAGE_SIZE,
            )

            x = image.img_to_array(img)

            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            feature = feature_extractor.predict(
                x,
                verbose=0,
            )

            features.append(feature.flatten())

            labels.append(cls)

    return np.array(features), np.array(labels)


print("Extracting Training Features...")
X_train, y_train = extract_features(TRAIN_DIR)

print("\nExtracting Validation Features...")
X_val, y_val = extract_features(VAL_DIR)

print("\nExtracting Test Features...")
X_test, y_test = extract_features(TEST_DIR)

print("\nFeature Extraction Completed.\n")

# ----------------------------------------------------
# Encode Labels
# ----------------------------------------------------

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_val = encoder.transform(y_val)

y_test = encoder.transform(y_test)

# ----------------------------------------------------
# Scale Features
# ----------------------------------------------------

print("Scaling Features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)

X_test = scaler.transform(X_test)

print("Scaling Completed.\n")

# ----------------------------------------------------
# Train SVM
# ----------------------------------------------------

print("Training SVM...\n")

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
)

svm.fit(X_train, y_train)

print("Training Completed.\n")

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

predictions = svm.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 60)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, predictions))

# ----------------------------------------------------
# Save Models
# ----------------------------------------------------

joblib.dump(
    svm,
    MODEL_DIR / "svm_classifier.pkl",
)

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl",
)

joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.pkl",
)

print("\nModels Saved Successfully.")

from plotting import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_metrics,
)

plot_class_distribution(y_train)

plot_confusion_matrix(
    y_test,
    predictions,
    encoder.classes_,
)

plot_metrics(
    y_test,
    predictions,
    encoder.classes_,
)