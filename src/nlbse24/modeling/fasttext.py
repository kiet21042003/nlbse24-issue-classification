"""P2 fastText models.

Two classifiers live here:

``FastTextClassifier``
    A thin adapter over Facebook's ``fasttext`` library. The dependency is optional and
    imported lazily, so the shared test suite and CI keep working without it.

``FastTextStyleLinearClassifier``
    A pure scikit-learn stand-in built from the same ingredients fastText uses - the
    hashing trick over word and subword n-grams followed by a linear softmax. It needs no
    extra dependency, so it always runs and doubles as an ablation of fastText's learned
    embedding layer.
"""

import importlib.metadata
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

from nlbse24.modeling.base import BaseIssueClassifier

LABEL_PREFIX = "__label__"


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    n_splits: int = 5
    text_fields: tuple[str, ...] = ("title", "body")


def _flatten(text: str) -> str:
    """Collapse all whitespace onto one line.

    ``compose_text`` joins the title and body with a blank line, but fastText's supervised
    file format is strictly one example per line, so the separator has to go.
    """

    flattened = " ".join(text.split())
    return flattened or "__empty__"


def _load_backend() -> Any:
    """Import the optional fastText backend with an actionable error message.

    Absolute imports mean this resolves to the third-party ``fasttext`` package rather
    than to this module, despite the shared name.
    """

    try:
        import fasttext
    except ImportError as error:
        raise ImportError(
            "the fastText backend is missing; install it with "
            "`pip install -r requirements-fasttext.txt`"
        ) from error
    return fasttext


def _backend_version() -> str | None:
    """Resolve the installed backend version.

    The module exposes no ``__version__``, and the wheel is published under two
    distribution names, so both are tried before giving up.
    """

    for distribution in ("fasttext", "fasttext-wheel"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@dataclass(frozen=True, slots=True)
class FastTextSettings:
    dim: int = 100
    lr: float = 0.5
    epoch: int = 25
    word_ngrams: int = 2
    min_count: int = 1
    minn: int = 3
    maxn: int = 6
    bucket: int = 200_000
    loss: str = "softmax"


@dataclass(frozen=True, slots=True)
class FastTextConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    model: FastTextSettings = FastTextSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "FastTextConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FastTextConfig":
        experiment = dict(payload.get("experiment", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])
        config = cls(
            experiment=ExperimentSettings(**experiment),
            model=FastTextSettings(**payload.get("model", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        if self.model.dim < 1:
            raise ValueError("dim must be positive")
        if self.model.epoch < 1:
            raise ValueError("epoch must be positive")
        if self.model.lr <= 0:
            raise ValueError("lr must be positive")
        if self.model.word_ngrams < 1:
            raise ValueError("word_ngrams must be at least 1")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FastTextClassifier(BaseIssueClassifier):
    """Supervised fastText, trained through a temporary file in the library's own format."""

    def __init__(self, config: FastTextConfig | None = None) -> None:
        self.config = config or FastTextConfig()
        self.config.validate()
        self._model: Any = None
        self._classes: tuple[str, ...] = ()
        self._backend_version: str | None = None

    @property
    def name(self) -> str:
        return "fasttext_supervised"

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("the fastText model has not been fitted yet")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")
        backend = _load_backend()
        self._backend_version = _backend_version()
        settings = self.config.model

        # fastText reads from disk, so the partition handed over by the runner is written
        # to a scratch file and removed again; no dataset file is ever touched here.
        directory = Path(tempfile.mkdtemp(prefix="nlbse24_fasttext_"))
        training_file = directory / "train.txt"
        try:
            lines = [
                f"{LABEL_PREFIX}{label} {_flatten(text)}"
                for text, label in zip(texts, labels, strict=True)
            ]
            training_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self._model = backend.train_supervised(
                input=str(training_file),
                dim=settings.dim,
                lr=settings.lr,
                epoch=settings.epoch,
                wordNgrams=settings.word_ngrams,
                minCount=settings.min_count,
                minn=settings.minn,
                maxn=settings.maxn,
                bucket=settings.bucket,
                loss=settings.loss,
                # fastText 0.9.2 exposes no seed; single-threaded training is what makes
                # a run reproducible.
                thread=1,
                verbose=0,
            )
        finally:
            training_file.unlink(missing_ok=True)
            directory.rmdir()

        self._classes = tuple(sorted(str(label) for label in set(labels)))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        self._check_fitted()
        flattened = [_flatten(text) for text in texts]
        predicted, _ = self._model.predict(flattened, k=1)
        return np.asarray(
            [str(row[0]).removeprefix(LABEL_PREFIX) for row in predicted], dtype=object
        )

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Probabilities over every label, reordered into ``classes_`` order."""

        self._check_fitted()
        flattened = [_flatten(text) for text in texts]
        predicted, probabilities = self._model.predict(flattened, k=-1)
        index = {label: position for position, label in enumerate(self._classes)}
        scores = np.zeros((len(flattened), len(self._classes)), dtype=float)
        for row, (labels, values) in enumerate(zip(predicted, probabilities, strict=True)):
            for label, value in zip(labels, values, strict=True):
                position = index.get(str(label).removeprefix(LABEL_PREFIX))
                if position is not None:
                    scores[row, position] = float(value)
        return scores

    @property
    def classes_(self) -> tuple[str, ...]:
        self._check_fitted()
        return self._classes

    def get_config(self) -> dict[str, Any]:
        payload = self.config.as_dict()
        # The shared artifact records versions for a fixed list of packages that cannot
        # include fastText, so the backend version is carried in the model config instead.
        payload["backend"] = {"library": "fasttext", "version": self._backend_version}
        payload["score_semantics"] = "probability"
        return payload

    # A native fastText handle cannot be pickled, so it is serialized through the
    # library's own format and the inherited joblib save/load keep working unchanged.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model_bytes"] = self._dump_model() if model is not None else None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        blob = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = self._load_model(blob) if blob is not None else None

    def _dump_model(self) -> bytes:
        directory = Path(tempfile.mkdtemp(prefix="nlbse24_fasttext_"))
        target = directory / "model.bin"
        try:
            self._model.save_model(str(target))
            return target.read_bytes()
        finally:
            target.unlink(missing_ok=True)
            directory.rmdir()

    @staticmethod
    def _load_model(blob: bytes) -> Any:
        backend = _load_backend()
        directory = Path(tempfile.mkdtemp(prefix="nlbse24_fasttext_"))
        target = directory / "model.bin"
        try:
            target.write_bytes(blob)
            return backend.load_model(str(target))
        finally:
            target.unlink(missing_ok=True)
            directory.rmdir()


@dataclass(frozen=True, slots=True)
class FastTextStyleSettings:
    n_features: int = 2**20
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    use_char_ngrams: bool = True
    alpha: float = 1e-5
    max_iter: int = 30
    tol: float = 1e-4
    class_weight: str | None = "balanced"


@dataclass(frozen=True, slots=True)
class FastTextStyleConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    model: FastTextStyleSettings = FastTextStyleSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "FastTextStyleConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FastTextStyleConfig":
        experiment = dict(payload.get("experiment", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])
        model = dict(payload.get("model", {}))
        for key in ("word_ngram_range", "char_ngram_range"):
            if key in model:
                model[key] = tuple(model[key])
        config = cls(
            experiment=ExperimentSettings(**experiment),
            model=FastTextStyleSettings(**model),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        if self.model.n_features < 2:
            raise ValueError("n_features must be greater than one")
        for label in ("word_ngram_range", "char_ngram_range"):
            start, end = getattr(self.model, label)
            if start < 1 or end < start:
                raise ValueError(f"invalid {label}")
        if self.model.alpha <= 0:
            raise ValueError("alpha must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FastTextStyleLinearClassifier(BaseIssueClassifier):
    """Hashed bag of word and subword n-grams with a linear softmax, in fastText's spirit."""

    def __init__(self, config: FastTextStyleConfig | None = None) -> None:
        self.config = config or FastTextStyleConfig()
        self.config.validate()
        settings = self.config.model
        # alternate_sign=False keeps the hashed counts non-negative, matching fastText's
        # bag-of-n-grams representation rather than a signed hashing sketch.
        branches: list[tuple[str, HashingVectorizer]] = [
            (
                "word",
                HashingVectorizer(
                    analyzer="word",
                    ngram_range=settings.word_ngram_range,
                    n_features=settings.n_features,
                    alternate_sign=False,
                    norm="l2",
                ),
            )
        ]
        if settings.use_char_ngrams:
            branches.append(
                (
                    "char",
                    HashingVectorizer(
                        analyzer="char_wb",
                        ngram_range=settings.char_ngram_range,
                        n_features=settings.n_features,
                        alternate_sign=False,
                        norm="l2",
                    ),
                )
            )
        self.pipeline = Pipeline(
            steps=[
                ("features", FeatureUnion(branches)),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        alpha=settings.alpha,
                        max_iter=settings.max_iter,
                        tol=settings.tol,
                        class_weight=settings.class_weight,
                        random_state=self.config.experiment.seed,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "fasttext_style_sgd"

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        check_is_fitted(self.pipeline)
        return self.pipeline.predict(list(texts))

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray:
        check_is_fitted(self.pipeline)
        return np.asarray(self.pipeline.predict_proba(list(texts)))

    @property
    def classes_(self) -> tuple[str, ...]:
        check_is_fitted(self.pipeline)
        estimator = self.pipeline.named_steps["classifier"]
        return tuple(str(label) for label in estimator.classes_)

    def get_config(self) -> dict[str, Any]:
        payload = self.config.as_dict()
        payload["score_semantics"] = "probability"
        return payload
