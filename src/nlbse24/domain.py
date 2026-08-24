"""Domain objects with no dependency on storage or model libraries."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nlbse24.constants import LABELS


class InvalidIssueRecord(ValueError):
    """Raised when an issue violates the shared data contract."""


@dataclass(frozen=True, slots=True)
class IssueRecord:
    """One labeled GitHub issue from the NLBSE'24 dataset."""

    repo: str
    created_at: str
    label: str
    title: str
    body: str

    def __post_init__(self) -> None:
        if not self.repo.strip():
            raise InvalidIssueRecord("repo must not be empty")
        if self.label not in LABELS:
            raise InvalidIssueRecord(f"unsupported label {self.label!r}; expected one of {LABELS}")
        if not self.title.strip():
            raise InvalidIssueRecord("title must not be empty")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "IssueRecord":
        """Build a record after a storage adapter has normalized missing values."""

        return cls(
            repo=str(row["repo"]),
            created_at=str(row["created_at"]),
            label=str(row["label"]),
            title=str(row["title"]),
            body=str(row.get("body", "")),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "repo": self.repo,
            "created_at": self.created_at,
            "label": self.label,
            "title": self.title,
            "body": self.body,
        }
