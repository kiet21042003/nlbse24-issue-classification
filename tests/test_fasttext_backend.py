"""End-to-end tests for the fastText adapter against a mocked backend.

The real ``fasttext`` wheel is an optional dependency that CI does not install, so the
adapter is driven here through a fake module injected into ``sys.modules``. The fake
deliberately returns labels in descending-probability order rather than class order, so
the reindexing in ``predict_scores`` is genuinely exercised.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from nlbse24.data import DatasetSplit, InMemoryIssueRepository
from nlbse24.modeling.fasttext import LABEL_PREFIX, FastTextClassifier, FastTextConfig
from nlbse24.runner import run_classifier_experiment

# Bodies deliberately contain the blank line that compose_text inserts between the title
# and the body; fastText's file format would break if it survived.
TEXTS = [
    "crash on open\n\nthe app dies with a segfault when opening a file",
    "startup exception\n\nbroken stack trace on startup",
    "null pointer\n\nthe application crashes with a null dereference",
    "dark mode\n\nrequest to add an option for dark mode",
    "plugin api\n\nproposal for a new plugin capability",
    "custom themes\n\nfeature request supporting custom themes",
    "build system\n\nhow do I configure the build system",
    "cli flags\n\nquestion about the usage of the cli flags",
    "install\n\nwhat is the recommended way to install this",
]
LABELS = ["bug"] * 3 + ["feature"] * 3 + ["question"] * 3


class _FakeFastTextModel:
    def __init__(self) -> None:
        self.memory: dict[str, str] = {}
        self.labels: list[str] = []
        self.kwargs: dict[str, object] = {}
        self.training_lines: list[str] = []

    def predict(self, texts: list[str], k: int = 1) -> tuple[list[tuple], list[np.ndarray]]:
        rows: list[tuple] = []
        scores: list[np.ndarray] = []
        for text in texts:
            best = self.memory.get(text, self.labels[0])
            others = [label for label in self.labels if label != best]
            # Highest probability first, which is not class order.
            ordered = [best, *others]
            values = [0.8, *[0.2 / len(others)] * len(others)] if others else [1.0]
            if k == 1:
                ordered, values = ordered[:1], values[:1]
            rows.append(tuple(ordered))
            scores.append(np.asarray(values))
        return rows, scores

    def save_model(self, path: str) -> None:
        payload = {"memory": self.memory, "labels": self.labels}
        Path(path).write_bytes(json.dumps(payload).encode("utf-8"))


def _train_supervised(*, input: str, **kwargs: object) -> _FakeFastTextModel:  # noqa: A002
    model = _FakeFastTextModel()
    model.kwargs = dict(kwargs)
    model.training_lines = Path(input).read_text(encoding="utf-8").splitlines()
    for line in model.training_lines:
        label, _, text = line.partition(" ")
        model.memory[text] = label
        if label not in model.labels:
            model.labels.append(label)
    model.labels.sort()
    return model


def _load_model(path: str) -> _FakeFastTextModel:
    payload = json.loads(Path(path).read_bytes().decode("utf-8"))
    model = _FakeFastTextModel()
    model.memory = dict(payload["memory"])
    model.labels = list(payload["labels"])
    return model


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fasttext")
    module.train_supervised = _train_supervised
    module.load_model = _load_model
    monkeypatch.setitem(sys.modules, "fasttext", module)


@pytest.fixture
def fitted(monkeypatch: pytest.MonkeyPatch) -> FastTextClassifier:
    install_fake_backend(monkeypatch)
    config = FastTextConfig.from_mapping({"model": {"dim": 10, "epoch": 3, "bucket": 1000}})
    return FastTextClassifier(config).fit(TEXTS, LABELS)


def test_contract_methods_fail_before_fitting() -> None:
    model = FastTextClassifier()

    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.predict(TEXTS)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.predict_scores(TEXTS)
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = model.classes_


def test_training_file_holds_exactly_one_line_per_example(fitted: FastTextClassifier) -> None:
    lines = fitted._model.training_lines

    assert len(lines) == len(TEXTS)
    assert all(line.startswith(LABEL_PREFIX) for line in lines)
    assert all("\n" not in line and "\r" not in line for line in lines)


def test_training_is_single_threaded_and_passes_no_seed(fitted: FastTextClassifier) -> None:
    kwargs = fitted._model.kwargs

    # fastText 0.9.2 accepts no seed argument; determinism comes from thread=1.
    assert kwargs["thread"] == 1
    assert "seed" not in kwargs
    assert kwargs["bucket"] == 1000
    assert kwargs["dim"] == 10


def test_predict_strips_the_label_prefix(fitted: FastTextClassifier) -> None:
    predictions = fitted.predict(TEXTS)

    assert fitted.classes_ == ("bug", "feature", "question")
    assert set(predictions) <= {"bug", "feature", "question"}
    assert list(predictions) == LABELS


def test_predict_scores_are_reindexed_into_class_order(fitted: FastTextClassifier) -> None:
    scores = fitted.predict_scores(TEXTS)

    assert scores.shape == (len(TEXTS), 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)
    highest = [fitted.classes_[index] for index in scores.argmax(axis=1)]
    assert highest == [str(label) for label in fitted.predict(TEXTS)]


def test_get_config_records_the_backend_and_score_semantics(fitted: FastTextClassifier) -> None:
    config = fitted.get_config()

    assert config["backend"]["library"] == "fasttext"
    assert config["score_semantics"] == "probability"
    assert config["model"]["bucket"] == 1000
    json.dumps(config)


def test_fit_rejects_mismatched_or_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_backend(monkeypatch)
    model = FastTextClassifier()

    with pytest.raises(ValueError, match="equal length"):
        model.fit(TEXTS, LABELS[:-1])
    with pytest.raises(ValueError, match="empty dataset"):
        model.fit([], [])


def test_native_model_survives_a_joblib_round_trip(
    fitted: FastTextClassifier, tmp_path: Path
) -> None:
    path = tmp_path / "fasttext.joblib"
    fitted.save(path)

    restored = FastTextClassifier.load(path)

    np.testing.assert_array_equal(restored.predict(TEXTS), fitted.predict(TEXTS))
    assert restored.classes_ == fitted.classes_


def test_an_unfitted_model_pickles_without_a_backend(tmp_path: Path) -> None:
    model = FastTextClassifier()
    path = tmp_path / "empty.joblib"

    model.save(path)
    restored = FastTextClassifier.load(path)

    assert restored.config == model.config


def test_fasttext_runs_through_the_shared_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_records: list
) -> None:
    install_fake_backend(monkeypatch)
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    config = FastTextConfig.from_mapping(
        {"experiment": {"seed": 7, "n_splits": 2}, "model": {"dim": 10, "bucket": 1000}}
    )

    def model_factory() -> FastTextClassifier:
        return FastTextClassifier(config)

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=model_factory,
        model_name="fasttext_supervised",
        output_dir=tmp_path,
        protocol="cv",
        seed=7,
        n_splits=2,
    )

    assert summary["evaluations"] == 4
    artifacts = list(tmp_path.glob("**/fold-*.json"))
    assert len(artifacts) == 4
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["classes"] == ["bug", "feature", "question"]
    assert len(payload["predictions"]["scores"][0]) == 3
