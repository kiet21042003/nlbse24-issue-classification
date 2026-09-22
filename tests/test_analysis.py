import json
from pathlib import Path

from nlbse24.evaluation.analysis import analyze_results, resource_summary


def _artifact(
    *,
    protocol: str = "cv",
    model: str = "demo",
    repository: str = "org/repo",
    fold: str = "fold-1",
    train_fingerprint: str = "train",
    test_fingerprint: str = "test",
    fit_seconds: float = 1.0,
) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": f"00000000-0000-0000-0000-{len(repository):012d}",
        "created_at_utc": "2026-09-21T00:00:00+00:00",
        "protocol": protocol,
        "fold": fold,
        "repository": repository,
        "seed": 42,
        "model": {"name": model, "config": {}},
        "data": {
            "upstream_commit": "abc",
            "train_size": 3,
            "test_size": 3,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": test_fingerprint,
        },
        "classes": ["bug", "feature", "question"],
        "metrics": {
            "per_class": {},
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
        },
        "resources": {
            "fit": {
                "elapsed_seconds": fit_seconds,
                "python_peak_mb": 2.0,
                "rss_delta_mb": 1.0,
            },
            "inference": {
                "elapsed_seconds": 0.1,
                "python_peak_mb": 1.0,
                "rss_delta_mb": 0.1,
            },
        },
        "predictions": {
            "y_true": ["bug", "feature", "question"],
            "y_pred": ["bug", "feature", "question"],
            "scores": None,
        },
        "environment": {},
    }


def test_analysis_writes_equal_repository_summaries(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for repository in ("a/repo", "b/repo"):
        path = root / "cv" / "demo" / repository.replace("/", "__") / "42"
        path.mkdir(parents=True)
        artifact = _artifact(repository=repository)
        (path / "fold-1.json").write_text(json.dumps(artifact), encoding="utf-8")

    tables = analyze_results(root)

    assert tables["evaluations"].shape[0] == 2
    assert tables["repositories"].shape[0] == 2
    assert tables["models"].iloc[0]["repositories"] == 2
    assert tables["models"].iloc[0]["macro_f1"] == 1.0


def test_resource_summary_counts_a_pooled_fit_once() -> None:
    first = _artifact(
        protocol="pooled_cv",
        repository="a/repo",
        train_fingerprint="pooled-train",
        fit_seconds=4.0,
    )
    second = _artifact(
        protocol="pooled_cv",
        repository="b/repo",
        train_fingerprint="pooled-train",
        fit_seconds=4.0,
    )

    summary = resource_summary([first, second]).iloc[0]

    assert summary["artifact_count"] == 2
    assert summary["fit_seconds_per_artifact"] == 8.0
    assert summary["fit_seconds_unique"] == 4.0
    assert summary["pooled_fit_reuse_factor"] == 2.0
