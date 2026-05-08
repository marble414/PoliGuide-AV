from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
except Exception:
    accuracy_score = None
    confusion_matrix = None
    precision_recall_fscore_support = None

from tpgr.data.labels import CANONICAL_GESTURES


def classification_report_dict(y_true: List[int], y_pred: List[int], class_names: List[str] | None = None) -> Dict[str, float]:
    class_names = class_names or CANONICAL_GESTURES
    kwargs = {"labels": list(range(len(class_names)))}
    if precision_recall_fscore_support is None or accuracy_score is None:
        precision, recall, f1 = _numpy_precision_recall_f1(y_true, y_pred, len(class_names))
        accuracy = _numpy_accuracy(y_true, y_pred)
    else:
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                zero_division=0,
                **kwargs,
            )
        except TypeError:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                **kwargs,
            )
        accuracy = float(accuracy_score(y_true, y_pred))
    macro = {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }
    for i, name in enumerate(class_names):
        macro[f"{name}_precision"] = float(precision[i])
        macro[f"{name}_recall"] = float(recall[i])
        macro[f"{name}_f1"] = float(f1[i])
    return macro


def _numpy_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(true == pred))


def _numpy_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= int(true_label) < num_classes and 0 <= int(pred_label) < num_classes:
            cm[int(true_label), int(pred_label)] += 1
    return cm


def _numpy_precision_recall_f1(y_true: List[int], y_pred: List[int], num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cm = _numpy_confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    tp = np.diag(cm)
    pred_count = cm.sum(axis=0)
    true_count = cm.sum(axis=1)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def save_confusion(y_true: List[int], y_pred: List[int], path_prefix: str | Path, class_names: List[str] | None = None) -> None:
    class_names = class_names or CANONICAL_GESTURES
    if confusion_matrix is None:
        cm = _numpy_confusion_matrix(y_true, y_pred, len(class_names))
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(path_prefix.with_suffix(".csv"), cm, delimiter=",", fmt="%d")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
