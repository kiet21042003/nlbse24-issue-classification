"""Tests for the protocol-coverage table and the missing-evaluation launcher."""

import pandas as pd

from scripts import protocol_coverage as coverage
from scripts import run_missing_evaluations as launcher


def _frame() -> pd.DataFrame:
    rows = []
    for repository, score in (("a/a", 0.8), ("b/b", 0.6)):
        for fold in ("fold-1", "fold-2"):
            rows.append(
                {
                    "model": "tfidf_logistic_regression",
                    "protocol": "cv",
                    "repository": repository,
                    "fold": fold,
                    "seed": 42,
                    "macro_f1": score,
                }
            )
    # a duplicate artifact must not be counted twice
    rows.append({**rows[0]})
    # a different seed must be ignored
    rows.append({**rows[0], "seed": 7, "macro_f1": 0.0})
    return pd.DataFrame(rows)


def test_coverage_counts_unique_evaluations_and_balances_repositories() -> None:
    table = coverage.coverage_table(_frame())
    row = next(item for item in table if item["model"] == "tfidf_logistic_regression")
    cv = row["cv"]
    assert cv["evaluations"] == 4
    assert cv["complete"] is False  # a complete CV run has 25 evaluations
    assert cv["macro_f1"] == 0.7  # mean of per-repository means 0.8 and 0.6
    assert row["official"]["evaluations"] == 0
    assert row["official"]["macro_f1"] is None


def test_markdown_and_latex_mark_missing_cells_without_imputing() -> None:
    table = coverage.coverage_table(_frame())
    markdown = coverage.render_markdown(table)
    assert "not run" in markdown
    assert "(partial 4/25)" in markdown
    latex = coverage.render_latex(table)
    assert "--" in latex
    assert r"\textbf{0.7000$^{*}$}" in latex


def test_launcher_orders_official_first_and_guards_the_official_test() -> None:
    results = launcher.Path("results")
    command = launcher.command_for(launcher.FROZEN[6], "official", results)
    assert "--confirm-official-test" in command
    assert command[command.index("--protocol") + 1] == "official"
    loo = launcher.command_for(launcher.FROZEN[6], "loo", results)
    assert "--confirm-official-test" not in loo
    assert launcher.PROTOCOLS[0] == "official"
