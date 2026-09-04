from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

def plot_class_distribution(labels):

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    plt.figure(figsize=(6,4))

    plt.bar(unique, counts)

    plt.title("Class Distribution")

    plt.ylabel("Images")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "class_distribution.png",
        dpi=300,
    )

    plt.close()

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax=ax)

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

from sklearn.metrics import precision_recall_fscore_support

def plot_metrics(
    y_true,
    y_pred,
    class_names,
):

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
    )

    x = np.arange(len(class_names))

    width = 0.25

    plt.figure(figsize=(8,5))

    plt.bar(
        x-width,
        precision,
        width,
        label="Precision",
    )

    plt.bar(
        x,
        recall,
        width,
        label="Recall",
    )

    plt.bar(
        x+width,
        f1,
        width,
        label="F1-score",
    )

    plt.xticks(
        x,
        class_names,
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "metrics.png",
        dpi=300,
    )

    plt.close()