import json
from pathlib import Path

import pytest

from nlbse24.evaluation.results import (
    artifact_to_row,
    find_artifact_paths,
    load_artifact,
)


def _artifact() -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "00000000-0000-0000-0000-000000000001",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "protocol": "cv",
        "fold": "fold-1",
        "repository": "org/repo",
        "seed": 42,
        "model": {"name": "demo", "config": {}},
        "data": {
            "upstream_commit": "abc",
            "train_size": 3,
            "test_size": 3,
            "train_fingerprint": "train",
            "test_fingerprint": "test",
        },
        "classes": ["bug", "feature", "question"],
        "metrics": {
            "per_class": {
                label: {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1-score": 1.0,
                    "support": 1.0,
                }
                for label in ("bug", "feature", "question")
            },
            "macro_average": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 3.0,
            },
            "weighted_average": {
                "precision": 1.0,
                "recall": 1.0,
                "f1-score": 1.0,
                "support": 3.0,
            },
            "accuracy": 1.0,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "resources": {
            "fit": {"elapsed_seconds": 1.5},
            "inference": {"elapsed_seconds": 0.2},
        },
        "predictions": {
            "y_true": ["bug", "feature", "question"],
            "y_pred": ["bug", "feature", "question"],
            "scores": None,
        },
        "environment": {},
    }


def test_load_artifact_and_flatten(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    artifact = load_artifact(path)
    row = artifact_to_row(artifact)

    assert row["model"] == "demo"
    assert row["macro_f1"] == 1.0
    assert row["fit_elapsed_seconds"] == 1.5


def test_find_artifact_paths_skips_summaries(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps(_artifact()), encoding="utf-8")
    (tmp_path / "summary-seed-42.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in find_artifact_paths(tmp_path)] == ["run.json"]


def test_load_artifact_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_artifact(path)
