"""Single source of truth for classification metrics."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nlbse24.constants import LABELS


def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("cannot evaluate empty predictions")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            metric: float(report[label][metric])
            for metric in ("precision", "recall", "f1-score", "support")
        }
        for label in LABELS
    }
    return {
        "per_class": per_class,
        "macro_average": {
            metric: float(report["macro avg"][metric])
            for metric in ("precision", "recall", "f1-score", "support")
        },
        "weighted_average": {
            metric: float(report["weighted avg"][metric])
            for metric in ("precision", "recall", "f1-score", "support")
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist(),
    }


def aggregate_macro_f1(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average folds within repositories, then repositories with equal weight."""

    if not rows:
        raise ValueError("cannot aggregate an empty result set")
    by_repository: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_repository[str(row["repository"])].append(float(row["macro_f1"]))
    repository_macro_f1 = {
        repo: float(np.mean(values)) for repo, values in sorted(by_repository.items())
    }
    return {
        "repository_macro_f1": repository_macro_f1,
        "cross_repository_macro_f1": float(np.mean(list(repository_macro_f1.values()))),
        "evaluations": len(rows),
    }
