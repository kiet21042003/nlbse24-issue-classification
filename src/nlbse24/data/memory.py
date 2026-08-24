"""In-memory persistence for tests, notebooks, and small experiments."""

from collections.abc import Mapping, Sequence

from nlbse24.data.base import DatasetSplit, IssueRepository
from nlbse24.domain import IssueRecord


class InMemoryIssueRepository(IssueRepository):
    def __init__(
        self,
        initial: Mapping[DatasetSplit, Sequence[IssueRecord]] | None = None,
    ) -> None:
        self._records: dict[DatasetSplit, list[IssueRecord]] = {
            split: list(records) for split, records in (initial or {}).items()
        }

    def load(self, split: DatasetSplit) -> list[IssueRecord]:
        return list(self._records.get(DatasetSplit.parse(split), []))

    def save(self, split: DatasetSplit, records: Sequence[IssueRecord]) -> None:
        self._records[DatasetSplit.parse(split)] = list(records)
