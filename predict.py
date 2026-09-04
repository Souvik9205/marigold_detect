from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# -----------------------------
# Paths
# -----------------------------

DATASET3 = Path("data/dataset3")

OUTPUT = Path("data/predicted")
OUTPUT.mkdir(exist_ok=True)

MODEL_DIR = Path("models")

IMAGE_SIZE = (224, 224)

# -----------------------------
# Load Models
# -----------------------------

print("Loading Models...")

svm = joblib.load(MODEL_DIR / "svm_classifier.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3),
)

feature_extractor = Model(
    inputs=base_model.input,
    outputs=GlobalAveragePooling2D()(base_model.output),
)

print("Models Loaded.\n")

rows = []

# -----------------------------
# Predict
# -----------------------------

images = list(DATASET3.glob("*"))

print(f"Found {len(images)} images\n")

for img_path in images:

    if img_path.suffix.lower() not in [".jpg",".jpeg",".png",".bmp"]:
        continue

    img = image.load_img(
        img_path,
        target_size=IMAGE_SIZE,
    )

    x = image.img_to_array(img)

    x = np.expand_dims(x,0)

    x = preprocess_input(x)

    feature = feature_extractor.predict(
        x,
        verbose=0,
    )

    feature = feature.flatten().reshape(1, -1)

    feature = scaler.transform(feature)

    print("Feature shape:", feature.shape)
    prediction = svm.predict(feature)[0]

    print(
        img_path.name,
        prediction,
        encoder.inverse_transform([prediction])[0]
    )

    probabilities = svm.decision_function(feature)

    label = encoder.inverse_transform([prediction])[0]

    folder = OUTPUT / label
    folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        img_path,
        folder / img_path.name,
    )

    rows.append({
        "filename": img_path.name,
        "predicted_label": label,
        "score": float(np.max(probabilities)),
    })

print("Saving CSV...")

df = pd.DataFrame(rows)

df.to_csv(
    "predictions.csv",
    index=False,
)

print(df.head())

print("\nPrediction Complete.")