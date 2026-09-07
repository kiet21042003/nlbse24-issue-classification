"""P3 RoBERTa experiment script: per-repository (via shared runner) and pooled
(standalone, P3-only ablation) training.
 
"""

from pathlib import Path
from typing import Any

from nlbse24.data import CsvIssueRepository, DatasetSplit, IssueDatasetService
from nlbse24.evaluation import (
    ResultWriter,
    aggregate_macro_f1,
    evaluate_predictions,
    make_run_artifact,
    profile_call,
)
from nlbse24.modeling import BaseIssueClassifier
from nlbse24.modeling.encoder import AdapterSettings, RobertaClassifier, RobertaConfig
from nlbse24.runner import run_classifier_experiment
from nlbse24.splits import stratified_repository_folds
from nlbse24.text import compose_texts


def _model_factory(config: RobertaConfig) -> BaseIssueClassifier:
    return RobertaClassifier(config)


def run_roberta_per_repository(
    *,
    config: RobertaConfig,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    repositories: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Per-repository training: independent model per repo, via the shared runner."""
    model_name = RobertaClassifier(config).name
    return run_classifier_experiment(
        repository=CsvIssueRepository(data_dir),
        model_factory=lambda: _model_factory(config),
        model_name=model_name,
        output_dir=output_dir,
        protocol="cv",
        seed=config.experiment.seed,
        n_splits=config.experiment.n_splits,
        text_fields=config.experiment.text_fields,
        repositories=repositories,
        overwrite=overwrite,
    )


def run_roberta_pooled(
    *,
    config: RobertaConfig,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "results",
    repositories: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pooled ablation: one model fit per fold on ALL repos' train rows pooled
    together, evaluated separately per repository. P3-only, not part of the
    shared runner's protocol set."""
    protocol = "pooled_cv"
    model_name = RobertaClassifier(config).name
    service = IssueDatasetService(CsvIssueRepository(data_dir))
    writer = ResultWriter(output_dir, overwrite=overwrite)
    text_fields = config.experiment.text_fields

    records = service.load(DatasetSplit.TRAIN, repositories)
    result_rows: list[dict[str, Any]] = []
    paths: list[str] = []

    for split in stratified_repository_folds(
        records, n_splits=config.experiment.n_splits, seed=config.experiment.seed
    ):
        train_records = [records[i] for i in split.train_indices]  # pooled: all repos
        model = _model_factory(config)
        train_texts = compose_texts(train_records, fields=text_fields)
        y_train = [record.label for record in train_records]
        _, fit_profile = profile_call(model.fit, train_texts, y_train)

        for repo in sorted({record.repo for record in records}):
            test_indices = [i for i in split.test_indices if records[i].repo == repo]
            if not test_indices:
                continue
            test_records = [records[i] for i in test_indices]
            test_texts = compose_texts(test_records, fields=text_fields)
            y_true = [record.label for record in test_records]

            y_pred_array, inference_profile = profile_call(model.predict, test_texts)
            y_pred = [str(label) for label in y_pred_array]
            scores = model.predict_scores(test_texts)
            metrics = evaluate_predictions(y_true, y_pred)

            artifact = make_run_artifact(
                protocol=protocol,
                fold=split.fold,
                repository=repo,
                seed=config.experiment.seed,
                model_name=model_name,
                model_config=model.get_config(),
                train_records=train_records,
                test_records=test_records,
                classes=model.classes_,
                metrics=metrics,
                resources={
                    "fit": fit_profile.as_dict(),
                    "inference": inference_profile.as_dict(),
                },
                y_true=y_true,
                y_pred=y_pred,
                scores=scores,
            )
            path = writer.write(artifact)
            paths.append(str(path))
            result_rows.append(
                {
                    "repository": repo,
                    "fold": split.fold,
                    "macro_f1": artifact["metrics"]["macro_average"]["f1-score"],
                }
            )

    aggregate = aggregate_macro_f1(result_rows)
    summary = {
        **aggregate,
        "split_strategy": "stratified_pooled_train_evaluated_per_repository",
        "text_fields": list(text_fields),
        "artifacts": paths,
    }
    summary_path = writer.write_summary(protocol, model_name, config.experiment.seed, summary)
    return {**summary, "summary_path": str(summary_path)}


def main() -> None:
    output_dir = Path("results/roberta")
    for adapter_enabled in (False, True):
        config = RobertaConfig(adapter=AdapterSettings(enabled=adapter_enabled))
        suffix = "adapter" if adapter_enabled else "full"

        per_repo = run_roberta_per_repository(config=config, output_dir=output_dir)
        print(f"per-repository ({suffix}):", per_repo)

        pooled = run_roberta_pooled(config=config, output_dir=output_dir)
        print(f"pooled ({suffix}):", pooled)


if __name__ == "__main__":
    main()