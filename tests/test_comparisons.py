import pytest

from nlbse24.evaluation.comparisons import group_comparable_artifacts, paired_bootstrap


def _artifact(model: str, repository: str = "org/repo") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": f"{model}-{repository}",
        "created_at_utc": "2026-09-21T00:00:00+00:00",
        "protocol": "cv",
        "fold": "fold-1",
        "repository": repository,
        "seed": 42,
        "model": {"name": model, "config": {}},
        "data": {
            "train_fingerprint": "train",
            "test_fingerprint": f"test-{repository}",
        },
        "classes": ["bug", "feature", "question"],
        "predictions": {
            "y_true": ["bug", "feature", "question", "bug", "feature", "question"],
            "y_pred": ["bug", "feature", "question", "bug", "feature", "question"],
            "scores": None,
        },
    }


def test_paired_bootstrap_is_deterministic_and_aligned() -> None:
    artifacts = [_artifact("a"), _artifact("b")]
    grouped = group_comparable_artifacts(artifacts, protocol="cv", models=["a", "b"])

    summary, differences = paired_bootstrap(grouped, n_resamples=50, seed=7)

    assert summary["point_estimate"].tolist() == [1.0, 1.0]
    assert differences.iloc[0]["point_difference"] == 0.0
    assert differences.iloc[0]["ci_low"] == 0.0
    assert differences.iloc[0]["ci_high"] == 0.0


def test_paired_bootstrap_rejects_mismatched_truth_rows() -> None:
    first = _artifact("a")
    second = _artifact("b")
    second["predictions"]["y_true"] = [
        "feature",
        "feature",
        "question",
        "bug",
        "feature",
        "question",
    ]

    with pytest.raises(ValueError, match="y_true rows are not aligned"):
        paired_bootstrap(
            group_comparable_artifacts([first, second], protocol="cv", models=["a", "b"]),
            n_resamples=10,
        )
