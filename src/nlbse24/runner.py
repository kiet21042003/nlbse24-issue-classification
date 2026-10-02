"""Shared experiment runner; model owners should reuse this evaluation path."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from nlbse24.data import CsvIssueRepository, DatasetSplit, IssueDatasetService, IssueRepository
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


def _fit_partition(
    *,
    train_records: list[IssueRecord],
    text_fields: tuple[str, ...],
    model_factory: ModelFactory,
) -> tuple[BaseIssueClassifier, Any]:
    """Fit a fresh model on one training partition and profile the fit."""

    model = model_factory()
    train_texts = compose_texts(train_records, fields=text_fields)
    y_train = [record.label for record in train_records]
    _, fit_profile = profile_call(model.fit, train_texts, y_train)
    return model, fit_profile


def _score_partition(
    *,
    model: BaseIssueClassifier,
    fit_profile: Any,
    train_records: list[IssueRecord],
    test_records: list[IssueRecord],
    protocol: str,
    fold: str,
    repository: str,
    seed: int,
    run_name: str,
    text_fields: tuple[str, ...],
    writer: ResultWriter,
) -> tuple[Path, dict[str, Any]]:
    """Evaluate an already fitted model on one test partition and write its artifact."""

    test_texts = compose_texts(test_records, fields=text_fields)
    y_true = [record.label for record in test_records]
    y_pred_array, inference_profile = profile_call(model.predict, test_texts)
    y_pred = [str(label) for label in y_pred_array]
    scores = model.predict_scores(test_texts)
    metrics = evaluate_predictions(y_true, y_pred)
    artifact = make_run_artifact(
        protocol=protocol,
        fold=fold,
        repository=repository,
        seed=seed,
        model_name=run_name,
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


def run_classifier_experiment(
    *,
    repository: IssueRepository,
    model_factory: ModelFactory,
    model_name: str,
    output_dir: str | Path = "results",
    protocol: str = "cv",
    seed: int = 42,
    n_splits: int = 5,
    text_fields: tuple[str, ...] = ("title", "body"),
    repositories: set[str] | None = None,
    overwrite: bool = False,
    allow_official_test: bool = False,
) -> dict[str, Any]:
    """Run any contract-compatible model through the shared experiment protocols."""

    service = IssueDatasetService(repository)
    writer = ResultWriter(output_dir, overwrite=overwrite)
    result_rows: list[dict[str, Any]] = []
    paths: list[str] = []

    def record(path: Path, artifact: dict[str, Any], fold: str, repository: str) -> None:
        paths.append(str(path))
        result_rows.append(
            {
                "repository": repository,
                "fold": fold,
                "macro_f1": artifact["metrics"]["macro_average"]["f1-score"],
            }
        )

    def evaluate(
        train_records: list[IssueRecord],
        test_records: list[IssueRecord],
        fold: str,
        repository: str,
    ) -> None:
        model, fit_profile = _fit_partition(
            train_records=train_records, text_fields=text_fields, model_factory=model_factory
        )
        path, artifact = _score_partition(
            model=model,
            fit_profile=fit_profile,
            train_records=train_records,
            test_records=test_records,
            protocol=protocol,
            fold=fold,
            repository=repository,
            seed=seed,
            run_name=model_name,
            text_fields=text_fields,
            writer=writer,
        )
        record(path, artifact, fold, repository)

    if protocol == "cv":
        split_strategy = "stratified_text_group_folds_per_repository"
        records = service.load(DatasetSplit.TRAIN, repositories)
        for split in stratified_repository_folds(records, n_splits=n_splits, seed=seed):
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
    elif protocol == "pooled_cv":
        # One model per fold is trained on every repository's training partition and then
        # evaluated separately on each repository's held-out partition (5 fits, 25 artifacts).
        # The fit profile is repeated across the repository artifacts of a fold.
        split_strategy = "stratified_pooled_train_evaluated_per_repository"
        records = service.load(DatasetSplit.TRAIN, repositories)
        for split in stratified_repository_folds(records, n_splits=n_splits, seed=seed):
            train_records = _select(records, split.train_indices)
            model, fit_profile = _fit_partition(
                train_records=train_records, text_fields=text_fields, model_factory=model_factory
            )
            for repo in sorted({item.repo for item in records}):
                test_indices = [
                    index for index in split.test_indices if records[index].repo == repo
                ]
                if not test_indices:
                    continue
                path, artifact = _score_partition(
                    model=model,
                    fit_profile=fit_profile,
                    train_records=train_records,
                    test_records=_select(records, test_indices),
                    protocol=protocol,
                    fold=split.fold,
                    repository=repo,
                    seed=seed,
                    run_name=model_name,
                    text_fields=text_fields,
                    writer=writer,
                )
                record(path, artifact, split.fold, repo)
    elif protocol == "loo":
        split_strategy = "leave_one_repository_out"
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
        split_strategy = "official_train_test_per_repository"
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
        "split_strategy": split_strategy,
        "text_fields": list(text_fields),
        "artifacts": paths,
    }
    summary_path = writer.write_summary(protocol, model_name, seed, summary)
    return {**summary, "summary_path": str(summary_path)}


def run_logistic_baseline(
    *,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    config_path: str | Path = "configs/logistic_regression.toml",
    protocol: str = "cv",
    repositories: set[str] | None = None,
    overwrite: bool = False,
    allow_official_test: bool = False,
    run_name: str = "tfidf_logistic_regression",
) -> dict[str, Any]:
    """Run P1's baseline under the same public runner available to P2-P5."""

    config = LogisticBaselineConfig.from_toml(config_path)

    def model_factory() -> BaseIssueClassifier:
        return TfidfLogisticRegressionClassifier(config)

    return run_classifier_experiment(
        repository=CsvIssueRepository(data_dir),
        model_factory=model_factory,
        model_name=run_name,
        output_dir=output_dir,
        protocol=protocol,
        seed=config.experiment.seed,
        n_splits=config.experiment.n_splits,
        text_fields=config.experiment.text_fields,
        repositories=repositories,
        overwrite=overwrite,
        allow_official_test=allow_official_test,
    )