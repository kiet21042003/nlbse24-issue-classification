"""P1 TF-IDF + Logistic Regression baseline."""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from nlbse24.modeling.base import BaseIssueClassifier


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    n_splits: int = 5
    text_fields: tuple[str, ...] = ("title", "body")


@dataclass(frozen=True, slots=True)
class VectorizerSettings:
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 1.0
    max_features: int | None = 50_000
    sublinear_tf: bool = True
    strip_accents: str | None = "unicode"


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    C: float = 4.0
    class_weight: str | None = "balanced"
    max_iter: int = 2_000
    solver: str = "lbfgs"


@dataclass(frozen=True, slots=True)
class LogisticBaselineConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    vectorizer: VectorizerSettings = VectorizerSettings()
    classifier: ClassifierSettings = ClassifierSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "LogisticBaselineConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LogisticBaselineConfig":
        experiment = dict(payload.get("experiment", {}))
        vectorizer = dict(payload.get("vectorizer", {}))
        classifier = dict(payload.get("classifier", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])
        if "ngram_range" in vectorizer:
            vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
        config = cls(
            experiment=ExperimentSettings(**experiment),
            vectorizer=VectorizerSettings(**vectorizer),
            classifier=ClassifierSettings(**classifier),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        start, end = self.vectorizer.ngram_range
        if start < 1 or end < start:
            raise ValueError("invalid ngram_range")
        if self.classifier.C <= 0:
            raise ValueError("classifier C must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TfidfLogisticRegressionClassifier(BaseIssueClassifier):
    def __init__(self, config: LogisticBaselineConfig | None = None) -> None:
        self.config = config or LogisticBaselineConfig()
        self.config.validate()
        vectorizer = self.config.vectorizer
        classifier = self.config.classifier
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer=vectorizer.analyzer,
                        ngram_range=vectorizer.ngram_range,
                        min_df=vectorizer.min_df,
                        max_df=vectorizer.max_df,
                        max_features=vectorizer.max_features,
                        sublinear_tf=vectorizer.sublinear_tf,
                        strip_accents=vectorizer.strip_accents,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=classifier.C,
                        class_weight=classifier.class_weight,
                        max_iter=classifier.max_iter,
                        solver=classifier.solver,
                        random_state=self.config.experiment.seed,
                    ),
                ),
            ]
        )

    @property
    def name(self) -> str:
        return "tfidf_logistic_regression"

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
        return self.pipeline.predict_proba(list(texts))

    @property
    def classes_(self) -> tuple[str, ...]:
        check_is_fitted(self.pipeline)
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        return tuple(str(label) for label in classifier.classes_)

    def get_config(self) -> dict[str, Any]:
        return self.config.as_dict()
