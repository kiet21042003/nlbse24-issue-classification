from pathlib import Path

from nlbse24.data import DatasetSplit, InMemoryIssueRepository
from nlbse24.modeling import LogisticBaselineConfig, TfidfLogisticRegressionClassifier
from nlbse24.runner import run_classifier_experiment


def test_generic_runner_writes_all_cv_artifacts(
    tmp_path: Path, sample_records: list
) -> None:
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    config = LogisticBaselineConfig.from_mapping(
        {
            "experiment": {"seed": 7, "n_splits": 2},
            "vectorizer": {"min_df": 1, "ngram_range": [1, 1]},
            "classifier": {"C": 1.0, "max_iter": 500},
        }
    )

    def model_factory() -> TfidfLogisticRegressionClassifier:
        return TfidfLogisticRegressionClassifier(config)

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=model_factory,
        model_name="tfidf_logistic_regression",
        output_dir=tmp_path,
        protocol="cv",
        seed=7,
        n_splits=2,
        overwrite=False,
    )
    assert summary["evaluations"] == 4
    assert len(summary["repository_macro_f1"]) == 2
    assert Path(summary["summary_path"]).is_file()
    assert len(list(tmp_path.glob("**/fold-*.json"))) == 4


def _logistic_factory() -> TfidfLogisticRegressionClassifier:
    config = LogisticBaselineConfig.from_mapping(
        {
            "experiment": {"seed": 7, "n_splits": 2},
            "vectorizer": {"min_df": 1, "ngram_range": [1, 1]},
            "classifier": {"C": 1.0, "max_iter": 500},
        }
    )
    return TfidfLogisticRegressionClassifier(config)


def test_pooled_cv_fits_once_per_fold_and_evaluates_every_repository(
    tmp_path: Path, sample_records: list
) -> None:
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    fits = 0

    def counting_factory() -> TfidfLogisticRegressionClassifier:
        nonlocal fits
        fits += 1
        return _logistic_factory()

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=counting_factory,
        model_name="tfidf_logistic_regression",
        output_dir=tmp_path,
        protocol="pooled_cv",
        seed=7,
        n_splits=2,
    )
    assert fits == 2  # one pooled model per fold, not one per repository
    assert summary["evaluations"] == 4  # 2 folds x 2 repositories
    assert summary["split_strategy"] == "stratified_pooled_train_evaluated_per_repository"
    artifacts = sorted(tmp_path.glob("pooled_cv/**/fold-*.json"))
    assert len(artifacts) == 4
    import json

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
    assert {payload["protocol"] for payload in payloads} == {"pooled_cv"}
    # the pooled training partition contains both repositories
    assert all(payload["data"]["train_size"] == 30 for payload in payloads)
    assert all(payload["data"]["test_size"] == 15 for payload in payloads)


def test_runner_rejects_unknown_protocol(tmp_path: Path, sample_records: list) -> None:
    import pytest

    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    with pytest.raises(ValueError, match="unsupported protocol"):
        run_classifier_experiment(
            repository=repository,
            model_factory=_logistic_factory,
            model_name="x",
            output_dir=tmp_path,
            protocol="nope",
        )
