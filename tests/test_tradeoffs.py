import pandas as pd

from nlbse24.evaluation.tradeoffs import pareto_frontier


def test_pareto_frontier_removes_dominated_rows() -> None:
    frame = pd.DataFrame(
        [
            {"model": "slow", "macro_f1": 0.80, "cost": 10.0},
            {"model": "fast", "macro_f1": 0.79, "cost": 2.0},
            {"model": "dominated", "macro_f1": 0.78, "cost": 12.0},
        ]
    )

    frontier = pareto_frontier(frame, cost_columns=("cost",))

    assert set(frontier["model"]) == {"slow", "fast"}


def test_pareto_frontier_never_compares_different_groups() -> None:
    frame = pd.DataFrame(
        [
            {"protocol": "cv", "model": "cv-model", "score": 0.80, "cost": 10.0},
            {"protocol": "pooled_cv", "model": "pooled-model", "score": 0.90, "cost": 1.0},
        ]
    )

    frontier = pareto_frontier(
        frame,
        score_column="score",
        cost_columns=("cost",),
        group_columns=("protocol",),
    )

    assert set(frontier["model"]) == {"cv-model", "pooled-model"}
