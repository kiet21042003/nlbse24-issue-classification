import json
from pathlib import Path

import pytest

from nlbse24.evaluation.results import (
    artifact_hardware_group,
    artifact_to_row,
    find_artifact_paths,
    load_artifact,
    results_dataframe,
)


def _artifact(*, protocol: str = "cv") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "00000000-0000-0000-0000-000000000001",
        "created_at_utc": "2026-08-29T00:00:00+00:00",
        "protocol": protocol,
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

    row = artifact_to_row(load_artifact(path))

    assert row["model"] == "demo"
    assert row["macro_f1"] == 1.0
    assert row["fit_elapsed_seconds"] == 1.5
    assert row["hardware_group"] == "unknown"


def test_find_artifact_paths_skips_summaries_and_analysis_json(tmp_path: Path) -> None:
    run_path = tmp_path / "cv" / "run.json"
    analysis_path = tmp_path / "analysis" / "comparison.json"
    summary_path = tmp_path / "summary-seed-42.json"
    run_path.parent.mkdir()
    analysis_path.parent.mkdir()
    run_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    analysis_path.write_text("{}", encoding="utf-8")
    summary_path.write_text("{}", encoding="utf-8")

    assert find_artifact_paths(tmp_path) == [run_path]


def test_load_artifact_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        load_artifact(path)


@pytest.mark.parametrize("protocol", ["cv", "loo", "official", "pooled_cv"])
def test_protocol_schema_and_ingestion_agree(tmp_path: Path, protocol: str) -> None:
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schemas/result.schema.json").read_text()
    )
    assert protocol in schema["properties"]["protocol"]["enum"]
    artifact = _artifact(protocol=protocol)
    (tmp_path / "fold-1.json").write_text(json.dumps(artifact), encoding="utf-8")
    frame = results_dataframe(tmp_path)
    assert frame["protocol"].tolist() == [protocol]
    assert frame["macro_f1"].tolist() == [1.0]


def test_hardware_group_accepts_explicit_artifact_identity() -> None:
    artifact = _artifact()
    artifact["environment"] = {"hardware_group": "gpu-a"}

    assert artifact_hardware_group(artifact) == "gpu-a"
