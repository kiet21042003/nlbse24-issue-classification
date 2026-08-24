"""Business logic that is independent of persistence format."""

from collections import Counter
from dataclasses import dataclass

from nlbse24.constants import LABELS, REPOSITORIES
from nlbse24.data.base import DatasetSplit, IssueRepository
from nlbse24.domain import IssueRecord


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    split: str
    total: int
    repositories: tuple[str, ...]
    labels: tuple[str, ...]
    counts: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "total": self.total,
            "repositories": list(self.repositories),
            "labels": list(self.labels),
            "counts": self.counts,
        }


class IssueDatasetService:
    def __init__(self, repository: IssueRepository) -> None:
        self.repository = repository

    def load(
        self, split: DatasetSplit, selected_repositories: set[str] | None = None
    ) -> list[IssueRecord]:
        records = self.repository.load(split)
        if selected_repositories is None:
            return records
        unknown = selected_repositories.difference(REPOSITORIES)
        if unknown:
            raise ValueError(f"unknown repositories: {sorted(unknown)}")
        return [record for record in records if record.repo in selected_repositories]

    def summarize(self, split: DatasetSplit) -> DatasetSummary:
        records = self.load(split)
        repositories = tuple(sorted({record.repo for record in records}))
        labels = tuple(sorted({record.label for record in records}))
        counter = Counter((record.repo, record.label) for record in records)
        counts = {
            repo: {label: counter[(repo, label)] for label in LABELS} for repo in repositories
        }
        return DatasetSummary(
            split=DatasetSplit.parse(split).value,
            total=len(records),
            repositories=repositories,
            labels=labels,
            counts=counts,
        )

    def validate_official_split(self, split: DatasetSplit) -> DatasetSummary:
        summary = self.summarize(split)
        errors: list[str] = []
        if summary.total != 1500:
            errors.append(f"expected 1500 rows, found {summary.total}")
        if summary.repositories != tuple(sorted(REPOSITORIES)):
            errors.append(f"unexpected repositories: {summary.repositories}")
        if summary.labels != tuple(sorted(LABELS)):
            errors.append(f"unexpected labels: {summary.labels}")
        for repo in REPOSITORIES:
            for label in LABELS:
                count = summary.counts.get(repo, {}).get(label, 0)
                if count != 100:
                    errors.append(f"expected 100 rows for {repo}/{label}, found {count}")
        if errors:
            raise ValueError("invalid official split: " + "; ".join(errors))
        return summary
