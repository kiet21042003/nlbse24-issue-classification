import pytest

from nlbse24.domain import IssueRecord


@pytest.fixture
def sample_records() -> list[IssueRecord]:
    records: list[IssueRecord] = []
    for repo in ("org/alpha", "org/beta"):
        for label in ("bug", "feature", "question"):
            for index in range(10):
                records.append(
                    IssueRecord(
                        repo=repo,
                        created_at="2024-01-01 00:00:00",
                        label=label,
                        title=f"{label} title {index}",
                        body=f"Body for {repo} {label} {index}",
                    )
                )
    return records
