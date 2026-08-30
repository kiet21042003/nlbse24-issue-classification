from pathlib import Path

import numpy as np
import pytest

from nlbse24.data import DatasetSplit, InMemoryIssueRepository
from nlbse24.modeling.sparse_models import (
    SparseModelConfig,
    TfidfSparseClassifier,
)
from nlbse24.runner import run_classifier_experiment

TEXTS = [
    "crash error failure segfault when opening a file",
    "broken exception error stack trace on startup",
    "the application crashes with a null pointer dereference",
    "request add an option for dark mode",
    "proposal for a new plugin api capability",
    "feature request support for custom themes",
    "how do I configure the build system",
    "question about the usage of the cli flags",
    "what is the recommended way to install this",
]
LABELS = ["bug"] * 3 + ["feature"] * 3 + ["question"] * 3

COMMITTED_CONFIGS = (
    "configs/tfidf_char_linear_svc.toml",
    "configs/tfidf_word_linear_svc.toml",
    "configs/tfidf_word_complement_nb.toml",
    "configs/ablations/p2_word_only.toml",
    "configs/ablations/p2_char_only.toml",
    "configs/ablations/p2_title_only.toml",
)


def small_config(kind: str, **overrides: object) -> SparseModelConfig:
    """A config small enough for a handful of documents."""

    payload: dict[str, object] = {
        "features": {"word": {"min_df": 1}, "char": {"min_df": 1, "ngram_range": [2, 3]}},
        "classifier": {"kind": kind},
    }
    payload.update(overrides)
    return SparseModelConfig.from_mapping(payload)


@pytest.mark.parametrize("kind", ["linear_svc", "complement_nb"])
def test_sparse_classifier_fits_predicts_and_serializes(kind: str, tmp_path: Path) -> None:
    model = TfidfSparseClassifier(small_config(kind)).fit(TEXTS, LABELS)

    predictions = model.predict(TEXTS)
    scores = model.predict_scores(TEXTS)

    assert predictions.shape == (9,)
    assert model.classes_ == ("bug", "feature", "question")
    assert scores.shape == (9, 3)

    path = tmp_path / "sparse.joblib"
    model.save(path)
    restored = TfidfSparseClassifier.load(path)
    np.testing.assert_array_equal(restored.predict(TEXTS), predictions)


@pytest.mark.parametrize("kind", ["linear_svc", "complement_nb"])
def test_score_columns_follow_class_order(kind: str) -> None:
    model = TfidfSparseClassifier(small_config(kind)).fit(TEXTS, LABELS)

    scores = model.predict_scores(TEXTS)
    highest = [model.classes_[index] for index in scores.argmax(axis=1)]

    assert highest == [str(label) for label in model.predict(TEXTS)]


def test_complement_nb_scores_are_probabilities() -> None:
    model = TfidfSparseClassifier(small_config("complement_nb")).fit(TEXTS, LABELS)

    np.testing.assert_allclose(model.predict_scores(TEXTS).sum(axis=1), 1.0)


def test_linear_svc_scores_are_unnormalised_margins() -> None:
    model = TfidfSparseClassifier(small_config("linear_svc")).fit(TEXTS, LABELS)

    scores = model.predict_scores(TEXTS)

    assert not np.allclose(scores.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("word", "char", "expected"),
    [
        (True, True, "tfidf_word_char_linear_svc"),
        (True, False, "tfidf_word_linear_svc"),
        (False, True, "tfidf_char_linear_svc"),
    ],
)
def test_name_reports_the_enabled_feature_branches(word: bool, char: bool, expected: str) -> None:
    config = SparseModelConfig.from_mapping(
        {
            "features": {
                "word": {"enabled": word, "min_df": 1},
                "char": {"enabled": char, "min_df": 1},
            }
        }
    )

    assert TfidfSparseClassifier(config).name == expected


def test_disabling_a_branch_removes_it_from_the_vocabulary() -> None:
    config = small_config("linear_svc", features={"word": {"enabled": False, "min_df": 1}})
    model = TfidfSparseClassifier(config).fit(TEXTS, LABELS)

    sizes = model.vocabulary_sizes()

    assert sizes is not None
    assert set(sizes) == {"char", "total"}
    assert sizes["total"] == sizes["char"]


def test_vocabulary_size_is_absent_before_fitting() -> None:
    model = TfidfSparseClassifier(small_config("linear_svc"))

    assert model.vocabulary_sizes() is None
    assert model.get_config()["fitted"] == {"vocabulary_size": None}


def test_get_config_records_the_fitted_vocabulary_size() -> None:
    model = TfidfSparseClassifier(small_config("linear_svc")).fit(TEXTS, LABELS)

    sizes = model.get_config()["fitted"]["vocabulary_size"]

    assert sizes["total"] == sizes["word"] + sizes["char"]


def test_non_positive_max_features_means_no_cap() -> None:
    config = SparseModelConfig.from_mapping(
        {"features": {"word": {"max_features": 0}, "char": {"max_features": -1}}}
    )

    assert config.features.word.max_features is None
    assert config.features.char.max_features is None


def test_empty_strip_accents_means_none() -> None:
    config = SparseModelConfig.from_mapping({"features": {"word": {"strip_accents": ""}}})

    assert config.features.word.strip_accents is None


def test_unknown_feature_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown feature settings"):
        SparseModelConfig.from_mapping({"features": {"word": {"nonsense": 1}}})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"experiment": {"n_splits": 1}}, "n_splits must be at least 2"),
        ({"experiment": {"text_fields": []}}, "text_fields must not be empty"),
        ({"classifier": {"kind": "random_forest"}}, "unsupported classifier kind"),
        (
            {"features": {"word": {"enabled": False}, "char": {"enabled": False}}},
            "at least one feature branch must be enabled",
        ),
        ({"features": {"char": {"ngram_range": [3, 2]}}}, "invalid ngram_range for the char"),
        ({"features": {"word": {"min_df": 0}}}, "min_df must be at least 1 for the word"),
        ({"classifier": {"linear_svc": {"C": 0}}}, "linear_svc C must be positive"),
        ({"classifier": {"complement_nb": {"alpha": 0}}}, "complement_nb alpha must be positive"),
    ],
)
def test_invalid_configuration_is_rejected(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SparseModelConfig.from_mapping(payload)


def test_fit_rejects_mismatched_or_empty_input() -> None:
    model = TfidfSparseClassifier(small_config("linear_svc"))

    with pytest.raises(ValueError, match="equal length"):
        model.fit(TEXTS, LABELS[:-1])
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit([], [])


@pytest.mark.parametrize("path", COMMITTED_CONFIGS)
def test_committed_sparse_configs_load(path: str) -> None:
    config = SparseModelConfig.from_toml(path)

    assert config.experiment.seed == 42
    assert config.experiment.n_splits == 5
    assert config.enabled_branches()
    assert TfidfSparseClassifier(config).name


def test_title_only_ablation_config_drops_the_body() -> None:
    config = SparseModelConfig.from_toml("configs/ablations/p2_title_only.toml")

    assert config.experiment.text_fields == ("title",)


def test_shared_runner_smoke_run_writes_all_cv_artifacts(
    tmp_path: Path, sample_records: list
) -> None:
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    config = SparseModelConfig.from_mapping(
        {
            "experiment": {"seed": 7, "n_splits": 2},
            "features": {"word": {"min_df": 1}, "char": {"min_df": 1, "ngram_range": [2, 3]}},
            "classifier": {"kind": "linear_svc"},
        }
    )

    def model_factory() -> TfidfSparseClassifier:
        return TfidfSparseClassifier(config)

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=model_factory,
        model_name="tfidf_word_char_linear_svc",
        output_dir=tmp_path,
        protocol="cv",
        seed=7,
        n_splits=2,
    )

    assert summary["evaluations"] == 4
    assert len(list(tmp_path.glob("**/fold-*.json"))) == 4
