"""End-to-end SetFit pipeline tests against a lightweight mocked backend."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from nlbse24.data import DatasetSplit, InMemoryIssueRepository
from nlbse24.modeling.setfit_classifier import SetFitClassifier, SetFitConfig
from nlbse24.runner import run_classifier_experiment


class _FakeDataset(dict):
    @classmethod
    def from_dict(cls, payload: dict) -> "_FakeDataset":
        return cls(payload)


class _FakeTrainingArguments:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _FakeSetFitModel:
    def __init__(self) -> None:
        self.model_body = types.SimpleNamespace(max_seq_length=None)
        self.memory: dict[str, int] = {}
        self.n_classes = 0

    @classmethod
    def from_pretrained(cls, backbone: str, **kwargs: object) -> "_FakeSetFitModel":
        if not backbone:
            raise ValueError("backbone must not be empty")
        instance = cls()
        instance.from_pretrained_kwargs = kwargs
        return instance

    def predict(self, texts: list[str]) -> list[int]:
        return [self.memory.get(text, 0) for text in texts]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        rows = []
        for predicted in self.predict(texts):
            row = np.full(self.n_classes, 0.1 / max(self.n_classes - 1, 1))
            row[int(predicted)] = 0.9
            rows.append(row)
        return np.asarray(rows)


class _FakeTrainer:
    def __init__(
        self,
        model: _FakeSetFitModel,
        args: _FakeTrainingArguments,
        train_dataset: _FakeDataset,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset

    def train(self) -> None:
        _FakeTrainer.last_args = dict(self.args.__dict__)
        labels = list(self.train_dataset["label"])
        self.model.n_classes = len(set(labels))
        for text, label in zip(self.train_dataset["text"], labels, strict=True):
            self.model.memory[text] = label


def install_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = _FakeDataset
    setfit_module = types.ModuleType("setfit")
    setfit_module.SetFitModel = _FakeSetFitModel
    setfit_module.TrainingArguments = _FakeTrainingArguments
    setfit_module.Trainer = _FakeTrainer
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "setfit", setfit_module)


TEXTS = [
    "crash error failure",
    "broken exception stack trace",
    "request add option",
    "proposal new capability",
    "how do I configure",
    "question help usage",
]
LABELS = ["bug", "bug", "feature", "feature", "question", "question"]


def test_fit_predict_scores_roundtrip_with_mocked_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    config = SetFitConfig.from_mapping(
        {
            "model": {"max_seq_length": 64},
            "sampling": {"num_epochs": 2, "max_examples_per_class": 2},
        }
    )
    model = SetFitClassifier(config).fit(TEXTS, LABELS)

    assert model.classes_ == ("bug", "feature", "question")
    assert model._model.model_body.max_seq_length == 64
    assert len(model._model.memory) == 6  # two shots per class survived sampling

    predictions = model.predict(TEXTS + ["totally unseen issue"])
    assert len(predictions) == len(TEXTS) + 1
    assert all(label in model.classes_ for label in predictions)
    assert predictions[-1] == "bug"  # unseen text falls back to class index 0

    scores = model.predict_scores(TEXTS[:4])
    assert scores.shape == (4, 3)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0)


def test_runner_end_to_end_with_mocked_setfit(
    tmp_path: Path,
    sample_records: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    config = SetFitConfig.from_mapping(
        {"experiment": {"seed": 7, "n_splits": 2}, "model": {"backbone": "stub/backbone"}}
    )

    def model_factory() -> SetFitClassifier:
        return SetFitClassifier(config)

    summary = run_classifier_experiment(
        repository=repository,
        model_factory=model_factory,
        model_name=SetFitClassifier(config).name,
        output_dir=tmp_path,
        protocol="cv",
        seed=7,
        n_splits=2,
        overwrite=True,
    )

    assert summary["evaluations"] == 4
    assert len(summary["repository_macro_f1"]) == 2
    assert all(0.0 <= value <= 1.0 for value in summary["repository_macro_f1"].values())
    assert Path(summary["summary_path"]).is_file()
    assert len(list(tmp_path.glob("**/fold-*.json"))) == 4


def test_mocked_backend_receives_truncation_and_sampling_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    config = SetFitConfig.from_mapping(
        {
            "model": {"backbone": "sentence-transformers/all-mpnet-base-v2", "device": "cpu"},
            "sampling": {"batch_size": 8, "num_iterations": 5},
        }
    )
    model = SetFitClassifier(config).fit(TEXTS, LABELS)
    assert model.name == "setfit_mpnet"
    assert model._model.from_pretrained_kwargs == {"device": "cpu"}


def test_official_baseline_arguments_are_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_backend(monkeypatch)
    config = SetFitConfig.from_toml("configs/setfit_mpnet.toml")
    model = SetFitClassifier(config).fit(TEXTS, LABELS)
    arguments = _FakeTrainer.last_args
    assert arguments["batch_size"] == (16, 2)
    assert arguments["num_iterations"] == 20
    assert arguments["num_epochs"] == 1
    assert arguments["seed"] == 42
    assert model._model.model_body.max_seq_length == 128
