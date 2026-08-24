from pathlib import Path

import pytest

from nlbse24.data import (
    CsvIssueRepository,
    DatasetSplit,
    InMemoryIssueRepository,
    JsonIssueRepository,
)
from nlbse24.domain import IssueRecord


@pytest.mark.parametrize("repository_type", [CsvIssueRepository, JsonIssueRepository])
def test_filesystem_repository_round_trip(
    tmp_path: Path,
    sample_records: list[IssueRecord],
    repository_type: type[CsvIssueRepository] | type[JsonIssueRepository],
) -> None:
    repository = repository_type(tmp_path)
    repository.save(DatasetSplit.TRAIN, sample_records)
    assert repository.load(DatasetSplit.TRAIN) == sample_records


def test_memory_repository_returns_a_copy(sample_records: list[IssueRecord]) -> None:
    repository = InMemoryIssueRepository({DatasetSplit.TRAIN: sample_records})
    first = repository.load(DatasetSplit.TRAIN)
    first.clear()
    assert repository.load(DatasetSplit.TRAIN) == sample_records


def test_csv_repository_normalizes_missing_body(tmp_path: Path) -> None:
    path = tmp_path / "issues_test.csv"
    path.write_text(
        "repo,created_at,label,title,body\norg/repo,2024-01-01,bug,Broken,\n",
        encoding="utf-8",
    )
    records = CsvIssueRepository(tmp_path).load(DatasetSplit.TEST)
    assert records[0].body == ""
