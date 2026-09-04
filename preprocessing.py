import os
import shutil
from pathlib import Path
import pandas as pd

import cv2
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")
TEST_DIR = Path("data/test")

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# -----------------------------
# Clean output folders
# -----------------------------
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Read dataset
# -----------------------------
image_paths = []
labels = []

print("=" * 50)
print("Scanning dataset...\n")

for class_dir in sorted(RAW_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    count = 0

    for img in class_dir.iterdir():
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        image_paths.append(str(img))
        labels.append(class_dir.name)
        count += 1

    print(f"{class_dir.name:<20}: {count}")

print(f"\nTotal Images : {len(image_paths)}")
print("=" * 50)

# -----------------------------
# Verify images
# -----------------------------
valid_paths = []
valid_labels = []

print("\nVerifying images...\n")

corrupted = 0

for path, label in zip(image_paths, labels):
    image = cv2.imread(path)

    if image is None:
        corrupted += 1
        print(f"Corrupted: {path}")
        continue

    valid_paths.append(path)
    valid_labels.append(label)

print(f"Valid Images     : {len(valid_paths)}")
print(f"Corrupted Images : {corrupted}")

# -----------------------------
# Train / Test Split
# -----------------------------
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    valid_paths,
    valid_labels,
    test_size=(1 - TRAIN_RATIO),
    stratify=valid_labels,
    random_state=RANDOM_STATE,
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels,
    random_state=RANDOM_STATE,
)

# -----------------------------
# Copy Function
# -----------------------------
def copy_images(paths, labels, destination):
    for img_path, label in zip(paths, labels):
        class_folder = destination / label
        class_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img_path, class_folder / Path(img_path).name)


copy_images(train_paths, train_labels, TRAIN_DIR)
copy_images(val_paths, val_labels, VAL_DIR)
copy_images(test_paths, test_labels, TEST_DIR)

# -----------------------------
# Print Statistics
# -----------------------------
def print_stats(name, labels):
    print(f"\n{name}")

    classes = sorted(set(labels))

    total = 0

    for c in classes:
        count = labels.count(c)
        total += count
        print(f"{c:<20}: {count}")

    print(f"{'Total':<20}: {total}")


print("\n" + "=" * 50)
print("Dataset Split Completed")
print("=" * 50)

print_stats("Training Set", train_labels)
print_stats("Validation Set", val_labels)
print_stats("Testing Set", test_labels)

# -----------------------------
# Save Split CSV
# -----------------------------
split_data = []

for path, label in zip(train_paths, train_labels):
    split_data.append({
        "filename": Path(path).name,
        "filepath": path,
        "label": label,
        "split": "train"
    })

for path, label in zip(val_paths, val_labels):
    split_data.append({
        "filename": Path(path).name,
        "filepath": path,
        "label": label,
        "split": "validation"
    })

for path, label in zip(test_paths, test_labels):
    split_data.append({
        "filename": Path(path).name,
        "filepath": path,
        "label": label,
        "split": "test"
    })

df = pd.DataFrame(split_data)

csv_path = Path("data") / "splits.csv"
df.to_csv(csv_path, index=False)

print(f"\nSplit information saved to: {csv_path}")

print("\nDone.")