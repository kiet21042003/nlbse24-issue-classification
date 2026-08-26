"""P4 SetFit sentence-transformer classifiers (MPNet and MiniLM variants)."""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from nlbse24.modeling.base import BaseIssueClassifier

_BACKBONE_SHORT_NAMES = {
    "sentence-transformers/all-mpnet-base-v2": "mpnet",
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
}


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    n_splits: int = 5
    text_fields: tuple[str, ...] = ("title", "body")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    backbone: str = "sentence-transformers/all-mpnet-base-v2"
    max_seq_length: int = 128
    device: str | None = None


@dataclass(frozen=True, slots=True)
class SamplingSettings:
    num_iterations: int = 20
    num_epochs: int = 1
    batch_size: int = 16
    head_batch_size: int | None = None
    max_examples_per_class: int | None = None


@dataclass(frozen=True, slots=True)
class SetFitConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    model: ModelSettings = ModelSettings()
    sampling: SamplingSettings = SamplingSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "SetFitConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SetFitConfig":
        experiment = dict(payload.get("experiment", {}))
        model = dict(payload.get("model", {}))
        sampling = dict(payload.get("sampling", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])
        config = cls(
            experiment=ExperimentSettings(**experiment),
            model=ModelSettings(**model),
            sampling=SamplingSettings(**sampling),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        if not self.model.backbone:
            raise ValueError("backbone must not be empty")
        if self.model.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.sampling.num_iterations < 1:
            raise ValueError("num_iterations must be at least 1")
        if self.sampling.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if self.sampling.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        per_class = self.sampling.max_examples_per_class
        if per_class is not None and per_class < 1:
            raise ValueError("max_examples_per_class must be at least 1 when set")
        if self.sampling.head_batch_size is not None and self.sampling.head_batch_size < 1:
            raise ValueError("head_batch_size must be at least 1 when set")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def subsample_indices_per_class(
    labels: Sequence[str], max_examples: int | None, seed: int
) -> list[int]:
    """Return indices keeping at most ``max_examples`` rows per class, seeded."""

    if max_examples is None or max_examples <= 0:
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    for label in sorted(set(labels)):
        candidates = np.flatnonzero(np.asarray(labels) == label)
        chosen = rng.choice(candidates, size=min(max_examples, len(candidates)), replace=False)
        kept.extend(int(index) for index in sorted(chosen))
    return sorted(kept)


class SetFitClassifier(BaseIssueClassifier):
    """SetFit contrastive fine-tuning of a sentence-transformer backbone.

    Truncation policy: inputs longer than ``max_seq_length`` tokens are
    tail-truncated by the underlying sentence-transformer; the 21k-word audit
    outlier therefore never reaches the encoder in full.
    """

    def __init__(self, config: SetFitConfig | None = None) -> None:
        self.config = config or SetFitConfig()
        self.config.validate()
        self._classes: tuple[str, ...] | None = None
        self._model: Any = None

    @property
    def name(self) -> str:
        short = _BACKBONE_SHORT_NAMES.get(
            self.config.model.backbone,
            self.config.model.backbone.rsplit("/", 1)[-1].lower().replace("-", "_"),
        )
        return f"setfit_{short}"

    @property
    def classes_(self) -> tuple[str, ...]:
        self._check_fitted()
        return self._classes

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")
        try:
            from datasets import Dataset as HFDataset
            from setfit import SetFitModel, Trainer, TrainingArguments
        except ImportError as error:
            raise ImportError(
                "SetFit extras are missing; install them with "
                "`pip install -r requirements-setfit.txt`"
            ) from error

        keep = subsample_indices_per_class(
            labels, self.config.sampling.max_examples_per_class, self.config.experiment.seed
        )
        self._classes = tuple(sorted(set(labels)))
        label_to_id = {label: index for index, label in enumerate(self._classes)}
        dataset = HFDataset.from_dict(
            {
                "text": [str(texts[index]) for index in keep],
                "label": [label_to_id[str(labels[index])] for index in keep],
            }
        )
        kwargs: dict[str, Any] = {}
        if self.config.model.device is not None:
            kwargs["device"] = self.config.model.device
        model = SetFitModel.from_pretrained(self.config.model.backbone, **kwargs)
        model.model_body.max_seq_length = self.config.model.max_seq_length
        batch_size: int | tuple[int, int] = self.config.sampling.batch_size
        if self.config.sampling.head_batch_size is not None:
            batch_size = (self.config.sampling.batch_size, self.config.sampling.head_batch_size)
        arguments = TrainingArguments(
            batch_size=batch_size,
            num_epochs=self.config.sampling.num_epochs,
            num_iterations=self.config.sampling.num_iterations,
            seed=self.config.experiment.seed,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
        trainer.train()
        self._model = model
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        self._check_fitted()
        predictions = self._model.predict(list(texts))
        return np.asarray([self._classes[int(index)] for index in predictions], dtype=object)

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        self._check_fitted()
        probabilities = self._model.predict_proba(list(texts))
        scores = np.asarray(probabilities, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != len(self._classes):
            raise RuntimeError(
                f"unexpected score shape {scores.shape}; expected (n, {len(self._classes)})"
            )
        return scores

    def get_config(self) -> dict[str, Any]:
        return self.config.as_dict()

    def _check_fitted(self) -> None:
        if self._model is None or self._classes is None:
            raise RuntimeError("SetFitClassifier is not fitted yet; call fit() first")
