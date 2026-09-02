import pytest

from nlbse24.evaluation.bootstrap import bootstrap_macro_f1


def test_bootstrap_is_deterministic_for_same_seed() -> None:
    y_true = ["bug", "bug", "feature", "feature", "question", "question"]
    y_pred = ["bug", "feature", "feature", "feature", "question", "bug"]

    first = bootstrap_macro_f1(y_true, y_pred, n_resamples=100, seed=7)
    second = bootstrap_macro_f1(y_true, y_pred, n_resamples=100, seed=7)

    assert first == second
    assert 0.0 <= first["ci_low"] <= first["ci_high"] <= 1.0


def test_bootstrap_perfect_predictions_have_perfect_ci() -> None:
    labels = ["bug", "feature", "question"] * 4

    result = bootstrap_macro_f1(labels, labels, n_resamples=100, seed=42)

    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["ci_high"] == pytest.approx(1.0)


def test_bootstrap_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        bootstrap_macro_f1([], [])
    with pytest.raises(ValueError):
        bootstrap_macro_f1(["bug"], ["bug", "feature"])
    with pytest.raises(ValueError):
        bootstrap_macro_f1(["bug"], ["bug"], n_resamples=0)