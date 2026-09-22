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
