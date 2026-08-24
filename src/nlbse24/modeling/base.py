"""Model contract shared by every team member."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np


class BaseIssueClassifier(ABC):
    """A storage-agnostic classifier operating on composed issue texts."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable, filesystem-safe model identifier."""

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        """Fit the model and return self."""

    @abstractmethod
    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return one string label for every text."""

    def predict_scores(self, texts: Sequence[str]) -> np.ndarray | None:
        """Return class-aligned scores when the model exposes them."""

        return None

    @property
    @abstractmethod
    def classes_(self) -> tuple[str, ...]:
        """Class order used by optional score columns."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return JSON-serializable model configuration."""

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"artifact at {path} is {type(model).__name__}, not {cls.__name__}")
        return model
