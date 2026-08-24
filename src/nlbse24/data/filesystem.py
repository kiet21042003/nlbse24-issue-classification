"""CSV and JSON implementations of the issue repository contract."""

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nlbse24.data.base import DatasetSplit, IssueRepository
from nlbse24.domain import IssueRecord

REQUIRED_COLUMNS = ("repo", "created_at", "label", "title", "body")


def _normalize_row(row: dict[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for column in REQUIRED_COLUMNS:
        value = row.get(column, "")
        normalized[column] = "" if value is None or pd.isna(value) else str(value)
    return normalized


class CsvIssueRepository(IssueRepository):
    """Persist official splits as `issues_train.csv` and `issues_test.csv`."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, split: DatasetSplit) -> Path:
        return self.directory / f"issues_{DatasetSplit.parse(split).value}.csv"

    def load(self, split: DatasetSplit) -> list[IssueRecord]:
        path = self.path_for(split)
        if not path.is_file():
            raise FileNotFoundError(f"dataset split not found: {path}")
        frame = pd.read_csv(path, keep_default_na=True)
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [IssueRecord.from_mapping(_normalize_row(row)) for row in frame.to_dict("records")]

    def save(self, split: DatasetSplit, records: Sequence[IssueRecord]) -> None:
        path = self.path_for(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        pd.DataFrame([record.as_dict() for record in records], columns=REQUIRED_COLUMNS).to_csv(
            temporary, index=False
        )
        temporary.replace(path)


class JsonIssueRepository(IssueRepository):
    """Human-readable filesystem persistence useful for demos and debugging."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def path_for(self, split: DatasetSplit) -> Path:
        return self.directory / f"issues_{DatasetSplit.parse(split).value}.json"

    def load(self, split: DatasetSplit) -> list[IssueRecord]:
        path = self.path_for(split)
        if not path.is_file():
            raise FileNotFoundError(f"dataset split not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        return [IssueRecord.from_mapping(_normalize_row(row)) for row in payload]

    def save(self, split: DatasetSplit, records: Sequence[IssueRecord]) -> None:
        path = self.path_for(split)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([record.as_dict() for record in records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
