import json
import random
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def sanitize_name(name: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in name
    )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        if prefer.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA was requested but is not available; using CPU.")
            return torch.device("cpu")
        return torch.device(prefer)
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint(state: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


# Label convention used throughout this module
#   0 = meteor   (POSITIVE class)
#   1 = non_meteor (negative class)
_POSITIVE_LABEL = 0  # meteor
_LABELS = [0, 1]
_CLASS_NAMES = ["Meteor", "Non-Meteor"]


def calculate_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    meteor_probabilities: Optional[Sequence[float]] = None,
) -> dict:
    """Compute binary classification metrics with meteor (label 0) as positive.

    Precision, recall, and F1 for meteor are exposed as top-level keys
    (``meteor_precision``, ``meteor_recall``, ``meteor_f1``) in addition to
    the ``per_class`` breakdown.

    ROC-AUC is computed with meteor as the positive class.  The sklearn
    ``roc_curve`` function treats 1 as positive by default; we avoid any
    ambiguity by converting: ``meteor_truth = (y_true == 0).astype(int)``,
    which maps meteor → 1 and non_meteor → 0 before passing to
    ``roc_curve``.  ``pos_label`` is therefore left at its default of 1.

    Warnings are emitted (not silenced) when:
    - Only one class is present in ``y_true``.
    - Any per-class metric is undefined (zero denominator).
    - Probability scores are missing, preventing ROC computation.
    """
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(predicted_labels, dtype=int)

    # ------------------------------------------------------------------ #
    # Validate class presence                                              #
    # ------------------------------------------------------------------ #
    present_true = set(np.unique(y_true).tolist())
    present_pred = set(np.unique(y_pred).tolist())
    missing_true = [lbl for lbl in _LABELS if lbl not in present_true]
    if missing_true:
        warnings.warn(
            f"calculate_metrics: true labels are missing class(es) "
            f"{missing_true}. Per-class metrics for those classes will be "
            f"zero. Macro averages may be misleading.",
            stacklevel=2,
        )
    missing_pred = [lbl for lbl in _LABELS if lbl not in present_pred]
    if missing_pred:
        warnings.warn(
            f"calculate_metrics: no samples were predicted as class(es) "
            f"{missing_pred}. Precision for those classes is undefined "
            f"and will be reported as 0.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------ #
    # Core per-class scores                                               #
    # ------------------------------------------------------------------ #
    matrix = confusion_matrix(y_true, y_pred, labels=_LABELS)
    support = matrix.sum(axis=1)  # actual counts per class

    precision_arr = precision_score(
        y_true, y_pred, labels=_LABELS, average=None, zero_division=0,
    )
    recall_arr = recall_score(
        y_true, y_pred, labels=_LABELS, average=None, zero_division=0,
    )
    f1_arr = f1_score(
        y_true, y_pred, labels=_LABELS, average=None, zero_division=0,
    )

    # Warn when a metric is 0 due to an empty denominator (undefined),
    # not just because the model failed to predict the class correctly.
    if 0 in missing_pred and int(support[0]) > 0:
        warnings.warn(
            "calculate_metrics: meteor precision is undefined "
            "(no predictions for meteor). Reported as 0.",
            stacklevel=2,
        )
    if 1 in missing_pred and int(support[1]) > 0:
        warnings.warn(
            "calculate_metrics: non_meteor precision is undefined "
            "(no predictions for non_meteor). Reported as 0.",
            stacklevel=2,
        )

    per_class = {}
    for index, class_name in enumerate(_CLASS_NAMES):
        per_class[class_name] = {
            "precision": float(precision_arr[index]),
            "recall": float(recall_arr[index]),
            "f1": float(f1_arr[index]),
            "support": int(support[index]),
        }

    # ------------------------------------------------------------------ #
    # Aggregate scores                                                     #
    # ------------------------------------------------------------------ #
    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        # --- meteor (positive class) top-level keys ---
        "meteor_precision": float(precision_arr[0]),
        "meteor_recall": float(recall_arr[0]),
        "meteor_f1": float(f1_arr[0]),
        # --- macro / weighted aggregates ---
        "precision_macro": float(
            precision_score(
                y_true, y_pred, labels=_LABELS,
                average="macro", zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred, labels=_LABELS,
                average="macro", zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred, labels=_LABELS,
                average="macro", zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true, y_pred, labels=_LABELS,
                average="weighted", zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, y_pred, labels=_LABELS,
                average="weighted", zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred, labels=_LABELS,
                average="weighted", zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=_LABELS,
            target_names=_CLASS_NAMES, zero_division=0,
        ),
        "roc_auc": None,
    }

    # ------------------------------------------------------------------ #
    # ROC / AUC — meteor is positive                                      #
    # ------------------------------------------------------------------ #
    if meteor_probabilities is None:
        warnings.warn(
            "calculate_metrics: no meteor probability scores provided; "
            "ROC-AUC cannot be computed.",
            stacklevel=2,
        )
        return metrics

    scores = np.asarray(meteor_probabilities, dtype=float)
    if len(scores) != len(y_true):
        warnings.warn(
            f"calculate_metrics: length mismatch between scores "
            f"({len(scores)}) and labels ({len(y_true)}); "
            "ROC-AUC skipped.",
            stacklevel=2,
        )
        return metrics

    if len(present_true) < 2:
        warnings.warn(
            "calculate_metrics: only one class present in true labels; "
            "ROC-AUC is undefined and will not be computed.",
            stacklevel=2,
        )
        return metrics

    # Recast so meteor (label 0) → 1 (positive) and non_meteor → 0.
    # sklearn roc_curve defaults pos_label=1; no override needed.
    meteor_truth = (y_true == _POSITIVE_LABEL).astype(int)
    try:
        fpr, tpr, thresholds = roc_curve(meteor_truth, scores)
        roc_auc_value = float(auc(fpr, tpr))
        metrics["roc_auc"] = roc_auc_value
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }
    except ValueError as exc:
        warnings.warn(
            f"calculate_metrics: ROC curve computation failed: {exc}",
            stacklevel=2,
        )

    return metrics


def save_json(data: dict, file_path: str) -> None:
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_jsonable(data), file, indent=4)


def save_metrics(metrics: dict, file_path: str) -> None:
    save_json(metrics, file_path)


def plot_history(
    history: dict,
    output_dir: str,
    prefix: str,
) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    paths = []

    if history.get("train_accuracy"):
        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs,
            history["train_accuracy"],
            label="Training Accuracy",
        )
        plt.plot(
            epochs,
            history["val_accuracy"],
            label="Validation Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        accuracy_path = output_path / f"{prefix}_accuracy.png"
        plt.savefig(accuracy_path)
        plt.close()
        paths.append(str(accuracy_path))

    if history.get("train_loss"):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history["train_loss"], label="Training Loss")
        plt.plot(epochs, history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_path = output_path / f"{prefix}_loss.png"
        plt.savefig(loss_path)
        plt.close()
        paths.append(str(loss_path))

    if history.get("train_f1_macro"):
        plt.figure(figsize=(8, 5))
        plt.plot(
            epochs,
            history["train_f1_macro"],
            label="Training Macro F1",
        )
        plt.plot(
            epochs,
            history["val_f1_macro"],
            label="Validation Macro F1",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title("Training and Validation Macro F1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        f1_path = output_path / f"{prefix}_f1.png"
        plt.savefig(f1_path)
        plt.close()
        paths.append(str(f1_path))

    return paths


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    output_path: str,
    class_names: Sequence[str] = ("Meteor", "Non-Meteor"),
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(values, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    threshold = values.max() / 2.0 if values.size else 0.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            plt.text(
                column,
                row,
                int(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_roc_curve(
    roc_curve_data: dict,
    roc_auc: Optional[float],
    output_path: str,
) -> None:
    if not roc_curve_data:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.plot(
        roc_curve_data["fpr"],
        roc_curve_data["tpr"],
        label=f"ROC AUC = {roc_auc:.4f}" if roc_auc is not None else "ROC",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Meteor as Positive Class)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, size: int = 1) -> None:
        self.total += float(value) * size
        self.count += size

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class EarlyStopping:
    def __init__(
        self,
        patience: int = 0,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.patience <= 0:
            return False

        if self.best is None:
            self.best = score
            return False

        if self.mode == "max":
            improved = score > self.best + self.min_delta
        else:
            improved = score < self.best - self.min_delta

        if improved:
            self.best = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
