<<<<<<< Updated upstream
import json
from pathlib import Path

import pytest

from nlbse24.evaluation.results import (
    artifact_to_row,
    find_artifact_paths,
    load_artifact,
)
=======
from nlbse24.evaluation.results import find_artifact_paths
>>>>>>> Stashed changes


def test_find_artifact_paths_ignores_derived_analysis_json(tmp_path):
    run_path = tmp_path / "cv" / "run.json"
    analysis_path = tmp_path / "analysis" / "comparison.json"
    run_path.parent.mkdir()
    analysis_path.parent.mkdir()
    run_path.write_text("{}", encoding="utf-8")
    analysis_path.write_text("{}", encoding="utf-8")

<<<<<<< Updated upstream

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
=======
    assert find_artifact_paths(tmp_path) == [run_path]
>>>>>>> Stashed changes
