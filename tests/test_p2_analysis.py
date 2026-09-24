"""Tests for the P2 per-class and bootstrap analysis script.

The script recomputes macro-F1 from confusion matrices instead of calling scikit-learn on
every bootstrap draw, which is the only reason a 10,000-resample paired bootstrap over
five models finishes in seconds. That shortcut is worth a test: if it ever stops agreeing
with the shared metrics module, every interval in the reports silently becomes wrong.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from sklearn.metrics import f1_score

from nlbse24.constants import LABELS

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_analysis() -> ModuleType:
    path = REPO_ROOT / "scripts" / "p2_analysis.py"
    spec = importlib.util.spec_from_file_location("p2_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["p2_analysis"] = module
    spec.loader.exec_module(module)
    return module


analysis = load_analysis()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_f1_from_counts_matches_scikit_learn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, len(LABELS), 240)
    y_pred = rng.integers(0, len(LABELS), 240)

    counts = analysis.confusion(y_true, y_pred)
    expected = f1_score(
        y_true,
        y_pred,
        labels=range(len(LABELS)),
        average=None,
        zero_division=0,
    )

    assert np.allclose(analysis.f1_from_counts(counts), expected)


def test_f1_from_counts_scores_a_stack_of_matrices_independently() -> None:
    rng = np.random.default_rng(7)
    stack = []
    expected = []
    for _ in range(16):
        y_true = rng.integers(0, len(LABELS), 60)
        y_pred = rng.integers(0, len(LABELS), 60)
        stack.append(analysis.confusion(y_true, y_pred))
        expected.append(
            f1_score(y_true, y_pred, labels=range(len(LABELS)), average=None, zero_division=0)
        )

    assert np.allclose(analysis.f1_from_counts(np.array(stack)), np.array(expected))


def test_a_label_nobody_predicts_or_holds_scores_zero() -> None:
    """zero_division=0 is what the shared metrics module uses, so match it exactly."""

    counts = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    counts[0, 0] = 10
    counts[1, 1] = 10

    scores = analysis.f1_from_counts(counts)

    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == 0.0


def test_cross_repository_f1_averages_folds_then_repositories() -> None:
    """The project averages evaluations inside a repository, never pools their rows.

    Pooling would weight a repository by its row count; averaging keeps the five
    repositories equal, which is what the competition metric and
    ``aggregate_macro_f1`` both do.
    """

    perfect = (np.array([0, 1, 2]), np.array([0, 1, 2]))
    wrong = (np.array([0, 1, 2]), np.array([1, 2, 0]))
    folds = {
        analysis.REPOSITORIES[0]: [perfect, wrong],
        analysis.REPOSITORIES[1]: [perfect, perfect],
        analysis.REPOSITORIES[2]: [perfect, perfect],
        analysis.REPOSITORIES[3]: [perfect, perfect],
        analysis.REPOSITORIES[4]: [perfect, perfect],
    }

    # The one bad fold drops its repository to 0.5, and that repository is one of five.
    assert analysis.cross_repository_f1(folds).mean() == pytest.approx(0.9, abs=1e-12)
