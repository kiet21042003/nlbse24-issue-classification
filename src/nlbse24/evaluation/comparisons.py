"""Paired comparisons over versioned result artifacts."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nlbse24.constants import LABELS

Artifact = dict[str, Any]
ArtifactKey = tuple[str, str, str, int, str]


def artifact_key(artifact: Artifact) -> ArtifactKey:
    """Return the identity used to align predictions from different models."""

    return (
        str(artifact["protocol"]),
        str(artifact["repository"]),
        int(artifact["seed"]),
        str(artifact["fold"]),
        str(artifact["data"]["test_fingerprint"]),
    )


def group_comparable_artifacts(
    artifacts: Iterable[Artifact],
    *,
    protocol: str,
    models: Sequence[str] | None = None,
    seed: int | None = None,
) -> dict[str, list[Artifact]]:
    """Group artifacts by model and reject incomplete comparison inputs."""

    selected = [
        artifact
        for artifact in artifacts
        if artifact["protocol"] == protocol and (seed is None or artifact["seed"] == seed)
    ]
    grouped: dict[str, list[Artifact]] = defaultdict(list)
    for artifact in selected:
        grouped[str(artifact["model"]["name"])].append(artifact)
    if models is not None:
        missing = sorted(set(models) - grouped.keys())
        if missing:
            raise ValueError(f"models have no {protocol!r} artifacts: {missing}")
        grouped = {model: grouped[model] for model in models}
    return {model: sorted(items, key=artifact_key) for model, items in grouped.items()}


def _check_alignment(grouped: Mapping[str, Sequence[Artifact]]) -> list[ArtifactKey]:
    if not grouped:
        raise ValueError("at least one model is required")
    models = list(grouped)
    keys_by_model = {
        model: [artifact_key(item) for item in items] for model, items in grouped.items()
    }
    first_keys = keys_by_model[models[0]]
    if len(set(first_keys)) != len(first_keys):
        raise ValueError(f"duplicate evaluation keys for model {models[0]!r}")
    for model in models[1:]:
        keys = keys_by_model[model]
        if keys != first_keys:
            raise ValueError(f"artifact evaluations are not aligned for model {model!r}")
    for index, key in enumerate(first_keys):
        reference = grouped[models[0]][index]
        reference_true = reference["predictions"]["y_true"]
        reference_classes = tuple(reference["classes"])
        if reference_classes != LABELS:
            raise ValueError(
                f"artifact {key} has class order {reference_classes}, expected {LABELS}"
            )
        for model in models[1:]:
            artifact = grouped[model][index]
            if tuple(artifact["classes"]) != LABELS:
                raise ValueError(f"artifact {key} has invalid class order for model {model!r}")
            if artifact["predictions"]["y_true"] != reference_true:
                raise ValueError(f"y_true rows are not aligned for evaluation {key}")
    return first_keys


def _macro_f1_from_confusion(counts: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(counts, axis1=-2, axis2=-1)
    denominator = counts.sum(axis=-2) + counts.sum(axis=-1)
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    )
    return f1.mean(axis=-1)


def _confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    indices = {label: index for index, label in enumerate(LABELS)}
    true = np.asarray([indices[label] for label in y_true], dtype=int)
    pred = np.asarray([indices[label] for label in y_pred], dtype=int)
    codes = true * len(LABELS) + pred
    return np.bincount(codes, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))


def paired_bootstrap(
    grouped: Mapping[str, Sequence[Artifact]],
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired repository-balanced bootstrap intervals.

    Resamples are shared by all models for each aligned evaluation, making the
    model differences paired.  Fold scores are averaged within repositories,
    then repositories receive equal weight, matching ``aggregate_macro_f1``.
    """

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    keys = _check_alignment(grouped)
    models = list(grouped)
    by_model_and_key = {
        model: {artifact_key(item): item for item in items} for model, items in grouped.items()
    }
    repositories = sorted({key[1] for key in keys})
    if not repositories:
        raise ValueError("no evaluations supplied")

    rng = np.random.default_rng(seed)
    bootstrap = {model: np.zeros(n_resamples, dtype=float) for model in models}
    point_by_model: dict[str, list[float]] = {model: [] for model in models}
    for repository in repositories:
        repository_keys = [key for key in keys if key[1] == repository]
        per_repository = {model: np.zeros(n_resamples, dtype=float) for model in models}
        point_repository = {model: [] for model in models}
        for key in repository_keys:
            reference = by_model_and_key[models[0]][key]
            size = len(reference["predictions"]["y_true"])
            if size == 0:
                raise ValueError(f"empty prediction set for evaluation {key}")
            draws = rng.integers(0, size, size=(n_resamples, size))
            for model in models:
                artifact = by_model_and_key[model][key]
                y_true = artifact["predictions"]["y_true"]
                y_pred = artifact["predictions"]["y_pred"]
                point_counts = _confusion(y_true, y_pred)
                point_repository[model].append(
                    float(_macro_f1_from_confusion(point_counts[None])[0])
                )
                true_indices = np.asarray([LABELS.index(label) for label in y_true], dtype=int)
                pred_indices = np.asarray([LABELS.index(label) for label in y_pred], dtype=int)
                codes = true_indices * len(LABELS) + pred_indices
                sampled_codes = codes[draws]
                counts = np.zeros((n_resamples, len(LABELS), len(LABELS)), dtype=np.int64)
                for label_index in range(len(LABELS) ** 2):
                    counts.reshape(n_resamples, -1)[:, label_index] = np.sum(
                        sampled_codes == label_index, axis=1
                    )
                per_repository[model] += _macro_f1_from_confusion(counts)
        for model in models:
            per_repository[model] /= len(repository_keys)
            bootstrap[model] += per_repository[model] / len(repositories)
            point_by_model[model].append(float(np.mean(point_repository[model])))

    alpha = 1.0 - confidence_level
    summary_rows = []
    for model in models:
        values = bootstrap[model]
        summary_rows.append(
            {
                "model": model,
                "point_estimate": float(np.mean(point_by_model[model])),
                "ci_low": float(np.quantile(values, alpha / 2.0)),
                "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
                "confidence_level": confidence_level,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        )
    summary = pd.DataFrame(summary_rows)
    reference = models[0]
    difference_rows = []
    for model in models[1:]:
        difference = bootstrap[reference] - bootstrap[model]
        difference_rows.append(
            {
                "reference_model": reference,
                "comparison_model": model,
                "point_difference": float(difference.mean()),
                "ci_low": float(np.quantile(difference, alpha / 2.0)),
                "ci_high": float(np.quantile(difference, 1.0 - alpha / 2.0)),
                "probability_reference_better": float(np.mean(difference > 0)),
                "confidence_level": confidence_level,
                "n_resamples": n_resamples,
                "seed": seed,
            }
        )
    return summary, pd.DataFrame(difference_rows)
