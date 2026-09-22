import pytest

from nlbse24.evaluation.ensemble import evaluate_ensemble, hard_vote, soft_vote


def _artifact(model: str, predictions: list[str], scores=None) -> dict:
    return {
        "protocol": "cv",
        "repository": "org/repo",
        "seed": 42,
        "fold": "fold-1",
        "model": {"name": model},
        "data": {"test_fingerprint": "test"},
        "classes": ["bug", "feature", "question"],
        "predictions": {
            "y_true": ["bug", "feature", "question"],
            "y_pred": predictions,
            "scores": scores,
        },
    }


def test_hard_vote_is_deterministic_on_ties() -> None:
    artifacts = [
        _artifact("a", ["bug", "feature", "question"]),
        _artifact("b", ["feature", "feature", "bug"]),
    ]

    assert hard_vote(artifacts) == ["bug", "feature", "bug"]


def test_soft_vote_rejects_uncalibrated_scores() -> None:
    artifacts = [
        _artifact("a", ["bug", "feature", "question"], [[1, 0, 0]] * 3),
        _artifact("b", ["bug", "feature", "question"], [[1, 0, 0]] * 3),
    ]
    artifacts[1]["predictions"]["scores"][0] = [3.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="probability"):
        soft_vote(artifacts)


def test_evaluate_ensemble_uses_repository_balanced_metric() -> None:
    artifacts = [
        _artifact("a", ["bug", "feature", "question"]),
        _artifact("b", ["bug", "feature", "question"]),
    ]

    result = evaluate_ensemble([artifacts])

    assert len(result["evaluations"]) == 1
    assert result["cross_repository_macro_f1"] == 1.0
