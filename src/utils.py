
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(true_labels, predicted_labels):
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(
            true_labels, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            true_labels, predicted_labels, zero_division=0
        ),
        "f1_score": f1_score(
            true_labels, predicted_labels, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels
        ).tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            target_names=["Meteor", "Non-Meteor"],
            zero_division=0
        )
    }


def save_metrics(metrics, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)
