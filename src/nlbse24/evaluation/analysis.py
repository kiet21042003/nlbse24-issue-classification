"""Aggregation helpers for the shared experiment result artifacts.

The competition metric gives every repository equal weight.  These helpers
therefore aggregate folds inside each repository first, and repositories only
afterwards.  Resource summaries also distinguish a pooled fit from the
repository-level artifacts that reuse that fit.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from nlbse24.evaluation.results import (
    artifact_hardware_group,
    load_artifacts,
    results_dataframe,
)

_METRIC_COLUMNS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)
_RESOURCE_MEAN_COLUMNS = ("fit_elapsed_seconds", "inference_elapsed_seconds")
_RESOURCE_MAX_COLUMNS = ("fit_python_peak_mb", "inference_python_peak_mb")


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame.columns]


def repository_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Average evaluations within each repository for each protocol/model."""

    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if "hardware_group" not in frame.columns:
        frame["hardware_group"] = "unknown"
    group_keys = ["protocol", "hardware_group", "model", "repository"]
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "mean") for column in _available(_METRIC_COLUMNS, frame)
    }
    aggregations.update(
        {
            column: (column, "mean")
            for column in _available(_RESOURCE_MEAN_COLUMNS, frame)
        }
    )
    aggregations.update(
        {
            column: (column, "max")
            for column in _available(_RESOURCE_MAX_COLUMNS, frame)
        }
    )
    aggregations["evaluations"] = ("run_id", "count")
    result = frame.groupby(group_keys, as_index=False).agg(**aggregations)
    return result.sort_values(group_keys).reset_index(drop=True)


def model_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Average repository summaries with equal repository weight."""

    repositories = repository_summary(frame)
    if repositories.empty:
        return repositories
    group_keys = ["protocol", "hardware_group", "model"]
    aggregations: dict[str, Any] = {
        column: (column, "mean")
        for column in _available(
            (*_METRIC_COLUMNS, *_RESOURCE_MEAN_COLUMNS, *_RESOURCE_MAX_COLUMNS),
            repositories,
        )
    }
    aggregations["repositories"] = ("repository", "nunique")
    result = repositories.groupby(group_keys, as_index=False).agg(**aggregations)
    return result.sort_values(group_keys).reset_index(drop=True)


def resource_summary(artifacts: list[dict[str, Any]]) -> pd.DataFrame:
    """Summarize resource measurements without counting pooled fits repeatedly."""

    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        resources = artifact.get("resources", {})
        fit = resources.get("fit", {})
        inference = resources.get("inference", {})
        rows.append(
            {
                "protocol": artifact["protocol"],
                "model": artifact["model"]["name"],
                "hardware_group": artifact_hardware_group(artifact),
                "repository": artifact["repository"],
                "seed": artifact["seed"],
                "fold": artifact["fold"],
                "train_fingerprint": artifact["data"]["train_fingerprint"],
                "fit_elapsed_seconds": fit.get("elapsed_seconds"),
                "inference_elapsed_seconds": inference.get("elapsed_seconds"),
                "fit_python_peak_mb": fit.get("python_peak_mb"),
                "inference_python_peak_mb": inference.get("python_peak_mb"),
                "fit_rss_delta_mb": fit.get("rss_delta_mb"),
                "inference_rss_delta_mb": inference.get("rss_delta_mb"),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # A pooled fit is reused by several repository artifacts.  The fingerprint
    # identifies the training partition, while protocol/model/seed/fold keeps
    # independent runs separate.
    fit_keys = [
        "protocol",
        "hardware_group",
        "model",
        "seed",
        "fold",
        "train_fingerprint",
    ]
    unique_fits = frame.drop_duplicates(fit_keys)
    grouped = frame.groupby(["protocol", "hardware_group", "model"], as_index=False)
    result = grouped.agg(
        artifact_count=("repository", "size"),
        repository_count=("repository", "nunique"),
        unique_fit_count=("train_fingerprint", "nunique"),
        fit_seconds_per_artifact=("fit_elapsed_seconds", "sum"),
        mean_fit_seconds=("fit_elapsed_seconds", "mean"),
        mean_inference_seconds=("inference_elapsed_seconds", "mean"),
        max_fit_python_peak_mb=("fit_python_peak_mb", "max"),
        max_inference_python_peak_mb=("inference_python_peak_mb", "max"),
        mean_fit_rss_delta_mb=("fit_rss_delta_mb", "mean"),
        mean_inference_rss_delta_mb=("inference_rss_delta_mb", "mean"),
    )
    unique_seconds = (
        unique_fits.groupby(
            ["protocol", "hardware_group", "model"], as_index=False
        )["fit_elapsed_seconds"]
        .sum()
        .rename(columns={"fit_elapsed_seconds": "fit_seconds_unique"})
    )
    result = result.merge(
        unique_seconds,
        on=["protocol", "hardware_group", "model"],
        how="left",
    )
    result["pooled_fit_reuse_factor"] = (
        result["fit_seconds_per_artifact"] / result["fit_seconds_unique"]
    ).where(result["fit_seconds_unique"] != 0)
    return result.sort_values(["protocol", "hardware_group", "model"]).reset_index(
        drop=True
    )


def analyze_results(root: str | Path, *, validate: bool = True) -> dict[str, pd.DataFrame]:
    """Load artifacts and return evaluation, repository, model and resource tables."""

    artifacts = load_artifacts(root, validate=validate)
    frame = results_dataframe(root, validate=validate)
    return {
        "evaluations": frame,
        "repositories": repository_summary(frame),
        "models": model_summary(frame),
        "resources": resource_summary(artifacts),
    }


def write_analysis_tables(
    root: str | Path, output_dir: str | Path, *, validate: bool = True
) -> dict[str, Path]:
    """Write CSV tables and return their paths."""

    tables = analyze_results(root, validate=validate)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
