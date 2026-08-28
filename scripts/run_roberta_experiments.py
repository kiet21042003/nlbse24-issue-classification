"""P3 RoBERTa experiment runner: per-repository versus pooled training.

Reuses the leakage-safe split protocols from splits.py; each fold is fed
into a fresh RobertaClassifier so full fine-tuning and adapter runs never
share weights across folds or repositories.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report

from nlbse24.domain import IssueRecord
from nlbse24.text import compose_text
from splits import SplitFold, leave_one_repository_out_folds, stratified_repository_folds

from roberta_classifier import AdapterSettings, RobertaClassifier, RobertaConfig

RESULT_METRICS = ("precision", "recall", "f1-score")


@dataclass(frozen=True, slots=True)
class FoldResult:
    protocol: str
    fold: str
    held_out_repository: str | None
    report: dict[str, dict[str, float]]


def _texts_and_labels(
    records: Sequence[IssueRecord], indices: Sequence[int]
) -> tuple[list[str], list[str]]:
    texts = [compose_text(records[index]) for index in indices]
    labels = [records[index].label for index in indices]
    return texts, labels


def run_fold(records: Sequence[IssueRecord], fold: SplitFold, config: RobertaConfig) -> FoldResult:
    """Fit one classifier on a fold's train split and score it on the test split."""

    train_texts, train_labels = _texts_and_labels(records, fold.train_indices)
    test_texts, test_labels = _texts_and_labels(records, fold.test_indices)

    classifier = RobertaClassifier(config)
    classifier.fit(train_texts, train_labels)
    predicted_labels = classifier.predict(test_texts)

    report = classification_report(test_labels, predicted_labels, digits=4, output_dict=True)
    return FoldResult(
        protocol=fold.protocol,
        fold=fold.fold,
        held_out_repository=fold.held_out_repository,
        report=report,
    )


def run_per_repository(records: Sequence[IssueRecord], config: RobertaConfig) -> list[FoldResult]:
    """Leave-one-repository-out: one classifier trained per held-out repository."""

    folds = leave_one_repository_out_folds(list(records))
    return [run_fold(records, fold, config) for fold in folds]


def run_pooled(records: Sequence[IssueRecord], config: RobertaConfig) -> list[FoldResult]:
    """Stratified cross-validation pooled across every repository."""

    folds = stratified_repository_folds(
        list(records), n_splits=config.experiment.n_splits, seed=config.experiment.seed
    )
    return [run_fold(records, fold, config) for fold in folds]


def summarize(results: list[FoldResult]) -> dict[str, float]:
    """Average the weighted-average precision/recall/f1 across folds."""

    per_fold_average = [result.report["weighted avg"] for result in results]
    return {
        metric: float(np.mean([row[metric] for row in per_fold_average]))
        for metric in RESULT_METRICS
    }


def save_results(results: list[FoldResult], output_path: Path) -> None:
    payload = [
        {
            "protocol": result.protocol,
            "fold": result.fold,
            "held_out_repository": result.held_out_repository,
            "report": result.report,
        }
        for result in results
    ]
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    # Replace with the team's actual loader (see the data module's contract).
    from nlbse24.data import load_issue_records

    records: list[IssueRecord] = load_issue_records()
    output_dir = Path("output/roberta")
    output_dir.mkdir(parents=True, exist_ok=True)

    for adapter_enabled in (False, True):
        config = RobertaConfig(adapter=AdapterSettings(enabled=adapter_enabled))
        suffix = "adapter" if adapter_enabled else "full"

        per_repo_results = run_per_repository(records, config)
        save_results(per_repo_results, output_dir / f"per_repository_{suffix}.json")
        print(f"per-repository ({suffix}):", summarize(per_repo_results))

        pooled_results = run_pooled(records, config)
        save_results(pooled_results, output_dir / f"pooled_{suffix}.json")
        print(f"pooled ({suffix}):", summarize(pooled_results))


if __name__ == "__main__":
    main()
