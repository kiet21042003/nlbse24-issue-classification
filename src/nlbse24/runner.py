"""Shared experiment runner; model owners should reuse this evaluation path."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nlbse24.data import CsvIssueRepository, DatasetSplit, IssueDatasetService
from nlbse24.domain import IssueRecord
from nlbse24.evaluation import (
    ResultWriter,
    aggregate_macro_f1,
    evaluate_predictions,
    make_run_artifact,
    profile_call,
)
from nlbse24.modeling import (
    BaseIssueClassifier,
    LogisticBaselineConfig,
    TfidfLogisticRegressionClassifier,
)
from nlbse24.splits import leave_one_repository_out_folds, stratified_repository_folds
from nlbse24.text import compose_texts

ModelFactory = Callable[[], BaseIssueClassifier]


def _select(records: Sequence[IssueRecord], indices: Sequence[int]) -> list[IssueRecord]:
    return [records[index] for index in indices]


def _evaluate_partition(
    *,
    train_records: list[IssueRecord],
    test_records: list[IssueRecord],
    protocol: str,
    fold: str,
    repository: str,
    seed: int,
    text_fields: tuple[str, ...],
    model_factory: ModelFactory,
    writer: ResultWriter,
) -> tuple[Path, dict[str, Any]]:
    model = model_factory()
    train_texts = compose_texts(train_records, fields=text_fields)
    test_texts = compose_texts(test_records, fields=text_fields)
    y_train = [record.label for record in train_records]
    y_true = [record.label for record in test_records]

    _, fit_profile = profile_call(model.fit, train_texts, y_train)
    y_pred_array, inference_profile = profile_call(model.predict, test_texts)
    y_pred = [str(label) for label in y_pred_array]
    scores = model.predict_scores(test_texts)
    metrics = evaluate_predictions(y_true, y_pred)
    artifact = make_run_artifact(
        protocol=protocol,
        fold=fold,
        repository=repository,
        seed=seed,
        model_name=model.name,
        model_config=model.get_config(),
        train_records=train_records,
        test_records=test_records,
        classes=model.classes_,
        metrics=metrics,
        resources={"fit": fit_profile.as_dict(), "inference": inference_profile.as_dict()},
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
    )
    return writer.write(artifact), artifact


def run_logistic_baseline(
    *,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    config_path: str | Path = "configs/logistic_regression.toml",
    protocol: str = "cv",
    repositories: set[str] | None = None,
    overwrite: bool = False,
    allow_official_test: bool = False,
) -> dict[str, Any]:
    """Run P1's baseline under a shared, leakage-safe protocol."""

    config = LogisticBaselineConfig.from_toml(config_path)
    service = IssueDatasetService(CsvIssueRepository(data_dir))
    writer = ResultWriter(output_dir, overwrite=overwrite)
    seed = config.experiment.seed
    fields = config.experiment.text_fields
    def model_factory() -> BaseIssueClassifier:
        return TfidfLogisticRegressionClassifier(config)

    result_rows: list[dict[str, Any]] = []
    paths: list[str] = []

    def evaluate(
        train_records: list[IssueRecord],
        test_records: list[IssueRecord],
        fold: str,
        repository: str,
    ) -> None:
        path, artifact = _evaluate_partition(
            train_records=train_records,
            test_records=test_records,
            protocol=protocol,
            fold=fold,
            repository=repository,
            seed=seed,
            text_fields=fields,
            model_factory=model_factory,
            writer=writer,
        )
        paths.append(str(path))
        result_rows.append(
            {
                "repository": repository,
                "fold": fold,
                "macro_f1": artifact["metrics"]["macro_average"]["f1-score"],
            }
        )

    if protocol == "cv":
        records = service.load(DatasetSplit.TRAIN, repositories)
        for split in stratified_repository_folds(
            records, n_splits=config.experiment.n_splits, seed=seed
        ):
            for repo in sorted({record.repo for record in records}):
                train_indices = [
                    index for index in split.train_indices if records[index].repo == repo
                ]
                test_indices = [
                    index for index in split.test_indices if records[index].repo == repo
                ]
                evaluate(
                    _select(records, train_indices),
                    _select(records, test_indices),
                    split.fold,
                    repo,
                )
    elif protocol == "loo":
        records = service.load(DatasetSplit.TRAIN, repositories)
        for split in leave_one_repository_out_folds(records):
            assert split.held_out_repository is not None
            evaluate(
                _select(records, split.train_indices),
                _select(records, split.test_indices),
                split.fold,
                split.held_out_repository,
            )
    elif protocol == "official":
        if not allow_official_test:
            raise PermissionError(
                "official test evaluation is locked; freeze the configuration and "
                "explicitly confirm it"
            )
        train_records = service.load(DatasetSplit.TRAIN, repositories)
        test_records = service.load(DatasetSplit.TEST, repositories)
        for repo in sorted({record.repo for record in train_records}):
            train_repo = [record for record in train_records if record.repo == repo]
            test_repo = [record for record in test_records if record.repo == repo]
            evaluate(train_repo, test_repo, "final", repo)
    else:
        raise ValueError(f"unsupported protocol: {protocol}")

    aggregate = aggregate_macro_f1(result_rows)
    summary = {
        **aggregate,
        "text_fields": list(fields),
        "artifacts": paths,
    }
    summary_path = writer.write_summary(
        protocol, "tfidf_logistic_regression", seed, summary
    )
    return {**summary, "summary_path": str(summary_path)}
