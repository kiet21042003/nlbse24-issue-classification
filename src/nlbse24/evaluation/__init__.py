"""Metrics, profiling, and result artifacts."""

from nlbse24.evaluation.analysis import (
    analyze_results,
    model_summary,
    repository_summary,
    resource_summary,
    write_analysis_tables,
)
from nlbse24.evaluation.artifacts import ResultWriter, make_run_artifact
from nlbse24.evaluation.comparisons import (
    group_comparable_artifacts,
    paired_bootstrap,
)
from nlbse24.evaluation.ensemble import evaluate_ensemble, hard_vote, soft_vote
from nlbse24.evaluation.metrics import aggregate_macro_f1, evaluate_predictions
from nlbse24.evaluation.profiling import ResourceProfile, profile_call
from nlbse24.evaluation.tradeoffs import pareto_frontier

__all__ = [
    "ResourceProfile",
    "ResultWriter",
    "analyze_results",
    "aggregate_macro_f1",
    "evaluate_predictions",
    "evaluate_ensemble",
    "group_comparable_artifacts",
    "hard_vote",
    "make_run_artifact",
    "model_summary",
    "paired_bootstrap",
    "pareto_frontier",
    "profile_call",
    "repository_summary",
    "resource_summary",
    "soft_vote",
    "write_analysis_tables",
]
