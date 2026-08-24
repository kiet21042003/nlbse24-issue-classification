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
