"""Tests for the dependency-free half of the P2 fastText module."""

import sys
from pathlib import Path

import numpy as np
import pytest

from nlbse24.data import DatasetSplit, InMemoryIssueRepository
from nlbse24.modeling.fasttext import (
    FastTextConfig,
    FastTextStyleConfig,
    FastTextStyleLinearClassifier,
    _flatten,
    _load_backend,
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


def small_style_config(**overrides: object) -> FastTextStyleConfig:
    payload: dict[str, object] = {"model": {"n_features": 2**14, "max_iter": 50}}
    payload["model"] = {**payload["model"], **overrides}
    return FastTextStyleConfig.from_mapping(payload)


def test_flatten_puts_an_example_on_one_line() -> None:
    assert _flatten("a title\n\nthe  body\r\ncontinues") == "a title the body continues"


def test_flatten_substitutes_a_token_for_blank_text() -> None:
    assert _flatten("   \n  ") == "__empty__"


def test_missing_backend_raises_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes "import fasttext" fail the way an absent
    # installation would.
    monkeypatch.setitem(sys.modules, "fasttext", None)

    with pytest.raises(ImportError, match="requirements-fasttext.txt"):
        _load_backend()


def test_style_classifier_fits_predicts_and_serializes(tmp_path: Path) -> None:
    model = FastTextStyleLinearClassifier(small_style_config()).fit(TEXTS, LABELS)

    predictions = model.predict(TEXTS)
    scores = model.predict_scores(TEXTS)

    assert model.name == "fasttext_style_sgd"
    assert model.classes_ == ("bug", "feature", "question")
    assert scores.shape == (9, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)

    path = tmp_path / "style.joblib"
    model.save(path)
    restored = FastTextStyleLinearClassifier.load(path)
    np.testing.assert_array_equal(restored.predict(TEXTS), predictions)


def test_style_score_columns_follow_class_order() -> None:
    model = FastTextStyleLinearClassifier(small_style_config()).fit(TEXTS, LABELS)

    scores = model.predict_scores(TEXTS)
    highest = [model.classes_[index] for index in scores.argmax(axis=1)]

    assert highest == [str(label) for label in model.predict(TEXTS)]


def test_style_classifier_can_drop_subword_features() -> None:
    model = FastTextStyleLinearClassifier(small_style_config(use_char_ngrams=False))

    union = model.pipeline.named_steps["features"]

    assert [name for name, _ in union.transformer_list] == ["word"]


def test_style_classifier_reports_its_configuration() -> None:
    model = FastTextStyleLinearClassifier(small_style_config())

    config = model.get_config()

    assert config["model"]["n_features"] == 2**14
    assert config["experiment"]["seed"] == 42


def test_style_fit_rejects_mismatched_or_empty_input() -> None:
    model = FastTextStyleLinearClassifier(small_style_config())

    with pytest.raises(ValueError, match="equal length"):
        model.fit(TEXTS, LABELS[:-1])
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit([], [])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"experiment": {"n_splits": 1}}, "n_splits must be at least 2"),
        ({"experiment": {"text_fields": []}}, "text_fields must not be empty"),
        ({"model": {"n_features": 1}}, "n_features must be greater than one"),
        ({"model": {"word_ngram_range": [2, 1]}}, "invalid word_ngram_range"),
        ({"model": {"char_ngram_range": [0, 3]}}, "invalid char_ngram_range"),
        ({"model": {"alpha": 0}}, "alpha must be positive"),
    ],
)
def test_invalid_style_configuration_is_rejected(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FastTextStyleConfig.from_mapping(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"experiment": {"n_splits": 1}}, "n_splits must be at least 2"),
        ({"experiment": {"text_fields": []}}, "text_fields must not be empty"),
        ({"model": {"dim": 0}}, "dim must be positive"),
        ({"model": {"epoch": 0}}, "epoch must be positive"),
        ({"model": {"lr": 0}}, "lr must be positive"),
        ({"model": {"word_ngrams": 0}}, "word_ngrams must be at least 1"),
    ],
)
def test_invalid_fasttext_configuration_is_rejected(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FastTextConfig.from_mapping(payload)


def test_committed_fasttext_configs_load() -> None:
    supervised = FastTextConfig.from_toml("configs/fasttext_supervised.toml")
    style = FastTextStyleConfig.from_toml("configs/fasttext_style_sgd.toml")

    assert supervised.experiment.seed == 42
    assert supervised.experiment.n_splits == 5
    assert style.experiment.seed == 42
    assert style.model.word_ngram_range == (1, 2)


def test_style_classifier_runs_through_the_shared_runner(
    tmp_path: Path, sample_records: list
) -> None:
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    config = FastTextStyleConfig.from_mapping(
        {"experiment": {"seed": 7, "n_splits": 2}, "model": {"n_features": 2**14}}
    )

    def model_factory() -> FastTextStyleLinearClassifier:
        return FastTextStyleLinearClassifier(config)

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=model_factory,
        model_name="fasttext_style_sgd",
        output_dir=tmp_path,
        protocol="cv",
        seed=7,
        n_splits=2,
    )

    assert summary["evaluations"] == 4
    assert len(list(tmp_path.glob("**/fold-*.json"))) == 4
