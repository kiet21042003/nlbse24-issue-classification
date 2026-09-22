"""Leakage-safe ensemble utilities for aligned result artifacts."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nlbse24.constants import LABELS
from nlbse24.evaluation.comparisons import artifact_key
from nlbse24.evaluation.metrics import aggregate_macro_f1, evaluate_predictions

Artifact = dict[str, Any]


def _validate_group(artifacts: Sequence[Artifact]) -> None:
    if not artifacts:
        raise ValueError("an ensemble needs at least one artifact")
    keys = [artifact_key(artifact) for artifact in artifacts]
    if len(set(keys)) != 1:
        raise ValueError("ensemble artifacts must describe one aligned evaluation")
    if any(tuple(artifact["classes"]) != LABELS for artifact in artifacts):
        raise ValueError("all ensemble artifacts must use the shared class order")
    reference = artifacts[0]["predictions"]["y_true"]
    if any(artifact["predictions"]["y_true"] != reference for artifact in artifacts[1:]):
        raise ValueError("ensemble artifacts do not have aligned y_true rows")


def hard_vote(
    artifacts: Sequence[Artifact], *, weights: Mapping[str, float] | None = None
) -> list[str]:
    """Return deterministic weighted majority-vote predictions."""

    _validate_group(artifacts)
    if weights is None:
        weights = {}
    model_predictions = [artifact["predictions"]["y_pred"] for artifact in artifacts]
    model_names = [str(artifact["model"]["name"]) for artifact in artifacts]
    resolved_weights = [float(weights.get(name, 1.0)) for name in model_names]
    if any(weight < 0 for weight in resolved_weights) or not any(resolved_weights):
        raise ValueError("ensemble weights must be non-negative and not all zero")

    predictions: list[str] = []
    for row in zip(*model_predictions, strict=True):
        totals = Counter({label: 0.0 for label in LABELS})
        for label, weight in zip(row, resolved_weights, strict=True):
            if label not in LABELS:
                raise ValueError(f"unknown predicted label {label!r}")
            totals[label] += weight
        predictions.append(max(LABELS, key=lambda label: (totals[label], -LABELS.index(label))))
    return predictions


def soft_vote(
    artifacts: Sequence[Artifact], *, weights: Mapping[str, float] | None = None
) -> list[str]:
    """Vote over probability scores.

    Scores must already be probabilities aligned to ``LABELS``.  Decision
    margins from LinearSVC or arbitrary model logits are rejected instead of
    being averaged as if they were calibrated probabilities.
    """

    _validate_group(artifacts)
    if weights is None:
        weights = {}
    probability_arrays: list[np.ndarray] = []
    model_weights: list[float] = []
    for artifact in artifacts:
        scores = artifact["predictions"].get("scores")
        if scores is None:
            raise ValueError(f"model {artifact['model']['name']!r} has no probability scores")
        array = np.asarray(scores, dtype=float)
        if array.ndim != 2 or array.shape[1] != len(LABELS):
            raise ValueError("ensemble scores must have shape (n_examples, 3)")
        if np.any(array < -1e-8) or np.any(array > 1.0 + 1e-8):
            raise ValueError("soft voting requires probability scores in [0, 1]")
        if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("soft voting requires scores whose rows sum to 1")
        probability_arrays.append(array)
        model_weights.append(float(weights.get(str(artifact["model"]["name"]), 1.0)))
    if any(weight < 0 for weight in model_weights) or not any(model_weights):
        raise ValueError("ensemble weights must be non-negative and not all zero")
    combined = np.average(np.stack(probability_arrays), axis=0, weights=model_weights)
    return [LABELS[index] for index in np.argmax(combined, axis=1)]


def evaluate_ensemble(
    grouped_artifacts: Sequence[Sequence[Artifact]],
    *,
    strategy: str = "hard",
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate one ensemble across aligned evaluation artifacts."""

    if not grouped_artifacts:
        raise ValueError("no evaluations supplied")
    predictions_by_evaluation: list[dict[str, Any]] = []
    for artifacts in grouped_artifacts:
        predictions = (
            hard_vote(artifacts, weights=weights)
            if strategy == "hard"
            else soft_vote(artifacts, weights=weights)
        )
        y_true = artifacts[0]["predictions"]["y_true"]
        metrics = evaluate_predictions(y_true, predictions)
        predictions_by_evaluation.append(
            {
                "repository": artifacts[0]["repository"],
                "fold": artifacts[0]["fold"],
                "macro_f1": metrics["macro_average"]["f1-score"],
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": predictions,
            }
        )
    summary = aggregate_macro_f1(predictions_by_evaluation)
    return {
        **summary,
        "strategy": strategy,
        "models": [str(artifact["model"]["name"]) for artifact in grouped_artifacts[0]],
        "evaluations": predictions_by_evaluation,
    }
