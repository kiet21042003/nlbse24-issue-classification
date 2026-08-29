"""P2 sparse lightweight baselines: TF-IDF feature unions with LinearSVC or ComplementNB.

One configurable classifier covers both estimators and every feature-ablation variant
(word only, character only, word + character), so the ablation is expressed in
configuration instead of duplicated code.
"""

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from nlbse24.modeling.base import BaseIssueClassifier

CLASSIFIER_KINDS = ("linear_svc", "complement_nb")


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    seed: int = 42
    n_splits: int = 5
    text_fields: tuple[str, ...] = ("title", "body")


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    """One TF-IDF branch of the feature union."""

    enabled: bool = True
    analyzer: str = "word"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 1.0
    max_features: int | None = 50_000
    sublinear_tf: bool = True
    strip_accents: str | None = "unicode"
    lowercase: bool = True

    def build(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents=self.strip_accents,
            lowercase=self.lowercase,
        )


@dataclass(frozen=True, slots=True)
class FeatureSpace:
    """Word and character branches; disabling one is how a feature ablation is declared."""

    word: FeatureSettings = FeatureSettings()
    # char_wb keeps n-grams inside word boundaries, which suits the issue-template markers
    # ("### ", "Type: <b>", "- [ ]") that dominate these issue bodies.
    char: FeatureSettings = FeatureSettings(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=100_000
    )


@dataclass(frozen=True, slots=True)
class LinearSvcSettings:
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 5_000
    tol: float = 1e-4


@dataclass(frozen=True, slots=True)
class ComplementNbSettings:
    alpha: float = 0.3
    norm: bool = False


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    kind: str = "linear_svc"
    linear_svc: LinearSvcSettings = LinearSvcSettings()
    complement_nb: ComplementNbSettings = ComplementNbSettings()


def _optional_int(value: Any) -> int | None:
    """TOML has no null literal, so a non-positive max_features means "no cap"."""

    if value is None:
        return None
    number = int(value)
    return number if number > 0 else None


def _optional_str(value: Any) -> str | None:
    """An empty TOML string means "pass None to scikit-learn"."""

    if value is None or value == "":
        return None
    return str(value)


def _feature_from_mapping(payload: Mapping[str, Any], default: FeatureSettings) -> FeatureSettings:
    known = {item.name for item in fields(FeatureSettings)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown feature settings: {sorted(unknown)}")
    settings = {name: getattr(default, name) for name in known}
    settings.update(payload)
    settings["ngram_range"] = tuple(settings["ngram_range"])
    settings["max_features"] = _optional_int(settings["max_features"])
    settings["strip_accents"] = _optional_str(settings["strip_accents"])
    return FeatureSettings(**settings)


@dataclass(frozen=True, slots=True)
class SparseModelConfig:
    experiment: ExperimentSettings = ExperimentSettings()
    features: FeatureSpace = FeatureSpace()
    classifier: ClassifierSettings = ClassifierSettings()

    @classmethod
    def from_toml(cls, path: str | Path) -> "SparseModelConfig":
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SparseModelConfig":
        experiment = dict(payload.get("experiment", {}))
        if "text_fields" in experiment:
            experiment["text_fields"] = tuple(experiment["text_fields"])

        features = dict(payload.get("features", {}))
        defaults = FeatureSpace()
        space = FeatureSpace(
            word=_feature_from_mapping(features.get("word", {}), defaults.word),
            char=_feature_from_mapping(features.get("char", {}), defaults.char),
        )

        classifier = dict(payload.get("classifier", {}))
        estimators = ClassifierSettings(
            kind=str(classifier.get("kind", "linear_svc")),
            linear_svc=LinearSvcSettings(**classifier.get("linear_svc", {})),
            complement_nb=ComplementNbSettings(**classifier.get("complement_nb", {})),
        )

        config = cls(experiment=ExperimentSettings(**experiment), features=space,
                     classifier=estimators)
        config.validate()
        return config

    def validate(self) -> None:
        if self.experiment.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.experiment.text_fields:
            raise ValueError("text_fields must not be empty")
        if self.classifier.kind not in CLASSIFIER_KINDS:
            raise ValueError(f"unsupported classifier kind: {self.classifier.kind}")
        if not self.enabled_branches():
            raise ValueError("at least one feature branch must be enabled")
        for label, feature in (("word", self.features.word), ("char", self.features.char)):
            start, end = feature.ngram_range
            if start < 1 or end < start:
                raise ValueError(f"invalid ngram_range for the {label} branch")
            if isinstance(feature.min_df, bool) or not isinstance(feature.min_df, int):
                raise ValueError(
                    f"min_df for the {label} branch must be an integer document count; "
                    "a float would silently be read as a proportion"
                )
            if feature.min_df < 1:
                raise ValueError(f"min_df must be at least 1 for the {label} branch")
        if self.classifier.linear_svc.C <= 0:
            raise ValueError("linear_svc C must be positive")
        if self.classifier.complement_nb.alpha <= 0:
            raise ValueError("complement_nb alpha must be positive")

    def enabled_branches(self) -> tuple[str, ...]:
        names = [
            name
            for name in ("word", "char")
            if getattr(self.features, name).enabled
        ]
        return tuple(names)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TfidfSparseClassifier(BaseIssueClassifier):
    """TF-IDF word/character feature union followed by a linear or naive Bayes estimator."""

    def __init__(self, config: SparseModelConfig | None = None) -> None:
        self.config = config or SparseModelConfig()
        self.config.validate()
        # Both vectorizers live inside the pipeline so they are refitted on the training
        # rows of every fold, which is what the protocol's leakage rules require.
        branches = [
            (name, getattr(self.config.features, name).build())
            for name in self.config.enabled_branches()
        ]
        self.pipeline = Pipeline(
            steps=[
                ("features", FeatureUnion(branches)),
                ("classifier", self._build_estimator()),
            ]
        )

    def _build_estimator(self) -> LinearSVC | ComplementNB:
        settings = self.config.classifier
        if settings.kind == "linear_svc":
            return LinearSVC(
                C=settings.linear_svc.C,
                class_weight=settings.linear_svc.class_weight,
                max_iter=settings.linear_svc.max_iter,
                tol=settings.linear_svc.tol,
                random_state=self.config.experiment.seed,
            )
        # ComplementNB is deterministic and requires non-negative features, which TF-IDF gives.
        return ComplementNB(alpha=settings.complement_nb.alpha, norm=settings.complement_nb.norm)

    @property
    def name(self) -> str:
        branches = "_".join(self.config.enabled_branches())
        return f"tfidf_{branches}_{self.config.classifier.kind}"

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
        """Class-aligned scores.

        ComplementNB returns probabilities. LinearSVC has no ``predict_proba``, so it
        returns one-vs-rest decision margins instead. Columns follow ``classes_`` in both
        cases, but only the ComplementNB columns sum to one.
        """

        check_is_fitted(self.pipeline)
        if self.config.classifier.kind == "complement_nb":
            scores = np.asarray(self.pipeline.predict_proba(list(texts)))
        else:
            estimator = self.pipeline.named_steps["classifier"]
            features = self.pipeline.named_steps["features"].transform(list(texts))
            scores = np.asarray(estimator.decision_function(features))
        # A two-class fit would make decision_function collapse to one dimension, which
        # would silently produce an artifact that violates the shared result schema.
        expected = len(self.classes_)
        if scores.ndim != 2 or scores.shape[1] != expected:
            raise ValueError(f"expected scores of shape (n, {expected}), found {scores.shape}")
        return scores

    @property
    def classes_(self) -> tuple[str, ...]:
        check_is_fitted(self.pipeline)
        estimator = self.pipeline.named_steps["classifier"]
        return tuple(str(label) for label in estimator.classes_)

    def vocabulary_sizes(self) -> dict[str, int] | None:
        """Fitted vocabulary size per branch; the data audit asks sparse models to record it."""

        union = self.pipeline.named_steps["features"]
        sizes: dict[str, int] = {}
        for name, vectorizer in union.transformer_list:
            vocabulary = getattr(vectorizer, "vocabulary_", None)
            if vocabulary is None:
                return None
            sizes[name] = len(vocabulary)
        sizes["total"] = sum(sizes.values())
        return sizes

    @property
    def score_semantics(self) -> str:
        """What predict_scores returns, so P5 does not average margins with probabilities."""

        if self.config.classifier.kind == "complement_nb":
            return "probability"
        return "decision_margin"

    def get_config(self) -> dict[str, Any]:
        payload = self.config.as_dict()
        # The runner calls get_config() after fit, so the realised vocabulary size travels
        # with the result artifact and P5 can plot it against memory use.
        payload["fitted"] = {"vocabulary_size": self.vocabulary_sizes()}
        payload["score_semantics"] = self.score_semantics
        return payload
