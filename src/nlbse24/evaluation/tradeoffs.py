"""Cost/performance trade-off analysis."""

from collections.abc import Sequence

import pandas as pd


def pareto_frontier(
    frame: pd.DataFrame,
    *,
    score_column: str = "macro_f1",
    cost_columns: Sequence[str] = ("mean_fit_seconds", "mean_inference_seconds"),
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return rows not dominated on score (higher) and costs (lower).

    A row is dominated when another row is at least as good on every selected
    dimension and strictly better on at least one dimension.  When
    ``group_columns`` is provided, dominance is computed independently within
    each group; this prevents mixing protocols or hardware measurements.
    """

    columns = [score_column, *cost_columns]
    missing_group_columns = [column for column in group_columns if column not in frame.columns]
    if missing_group_columns:
        raise ValueError(f"missing Pareto group columns: {missing_group_columns}")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing Pareto columns: {missing}")
    if frame.empty:
        return frame.copy()

    if group_columns:
        frontiers = []
        for _, group in frame.groupby(list(group_columns), dropna=False, sort=True):
            frontiers.append(
                pareto_frontier(
                    group,
                    score_column=score_column,
                    cost_columns=cost_columns,
                )
            )
        return (
            pd.concat(frontiers, ignore_index=True)
            if frontiers
            else frame.iloc[0:0].copy()
        )

    values = frame[columns].astype(float).to_numpy()
    keep = []
    for index, candidate in enumerate(values):
        dominated = False
        for other_index, other in enumerate(values):
            if index == other_index:
                continue
            no_worse_score = other[0] >= candidate[0]
            no_worse_costs = all(
                other[position] <= candidate[position]
                for position in range(1, len(columns))
            )
            strictly_better = other[0] > candidate[0] or any(
                other[position] < candidate[position] for position in range(1, len(columns))
            )
            if no_worse_score and no_worse_costs and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(index)
    return frame.iloc[keep].reset_index(drop=True)
