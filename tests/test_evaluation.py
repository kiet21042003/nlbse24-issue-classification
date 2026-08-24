import json
from pathlib import Path

import pytest

from nlbse24.evaluation.artifacts import ResultWriter
from nlbse24.evaluation.metrics import aggregate_macro_f1, evaluate_predictions


def test_evaluate_predictions_uses_fixed_class_order() -> None:
    metrics = evaluate_predictions(
        ["bug", "feature", "question"], ["bug", "feature", "bug"]
    )
    assert list(metrics["per_class"]) == ["bug", "feature", "question"]
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_aggregate_gives_repositories_equal_weight() -> None:
    summary = aggregate_macro_f1(
        [
            {"repository": "a/repo", "macro_f1": 0.5},
            {"repository": "a/repo", "macro_f1": 0.7},
            {"repository": "b/repo", "macro_f1": 0.8},
        ]
    )
    assert summary["repository_macro_f1"] == {"a/repo": 0.6, "b/repo": 0.8}
    assert summary["cross_repository_macro_f1"] == pytest.approx(0.7)


def test_result_writer_is_atomic_and_refuses_silent_overwrite(tmp_path: Path) -> None:
    artifact = {
        "schema_version": "1.0",
        "protocol": "cv",
        "model": {"name": "demo"},
        "repository": "org/repo",
        "seed": 42,
        "fold": "fold-1",
    }
    writer = ResultWriter(tmp_path)
    path = writer.write(artifact)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol"] == "cv"
    assert path == tmp_path / "cv" / "demo" / "org__repo" / "42" / "fold-1.json"
    with pytest.raises(FileExistsError):
        writer.write(artifact)
