"""Storage-independent issue repository interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum

from nlbse24.domain import IssueRecord


class DatasetSplit(str, Enum):
    TRAIN = "train"
    TEST = "test"

    @classmethod
    def parse(cls, value: str | DatasetSplit) -> DatasetSplit:
        return value if isinstance(value, cls) else cls(value)


class IssueRepository(ABC):
    """Persistence contract used by dataset business logic."""

    @abstractmethod
    def load(self, split: DatasetSplit) -> list[IssueRecord]:
        """Load all records for a named official split."""

    @abstractmethod
    def save(self, split: DatasetSplit, records: Sequence[IssueRecord]) -> None:
        """Persist records for a named official split."""
