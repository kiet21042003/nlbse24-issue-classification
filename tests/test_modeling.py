from pathlib import Path

import numpy as np

from nlbse24.modeling import LogisticBaselineConfig, TfidfLogisticRegressionClassifier


def test_logistic_baseline_fits_predicts_and_serializes(tmp_path: Path) -> None:
    texts = [
        "crash error failure",
        "broken exception error",
        "request add option",
        "proposal new capability",
        "how do I configure",
        "question help usage",
    ]
    labels = ["bug", "bug", "feature", "feature", "question", "question"]
    config = LogisticBaselineConfig.from_mapping(
        {
            "vectorizer": {"min_df": 1, "ngram_range": [1, 1]},
            "classifier": {"C": 1.0, "max_iter": 500},
        }
    )
    model = TfidfLogisticRegressionClassifier(config).fit(texts, labels)
    predictions = model.predict(texts)
    scores = model.predict_scores(texts)
    assert predictions.shape == (6,)
    assert scores.shape == (6, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)

    path = tmp_path / "baseline.joblib"
    model.save(path)
    restored = TfidfLogisticRegressionClassifier.load(path)
    np.testing.assert_array_equal(restored.predict(texts), predictions)


def test_committed_baseline_config_loads() -> None:
    config = LogisticBaselineConfig.from_toml("configs/logistic_regression.toml")
    assert config.experiment.n_splits == 5
    assert config.experiment.text_fields == ("title", "body")
