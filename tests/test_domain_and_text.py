import pytest

from nlbse24.domain import InvalidIssueRecord, IssueRecord
from nlbse24.text import compose_text


def test_issue_record_rejects_unknown_label() -> None:
    with pytest.raises(InvalidIssueRecord, match="unsupported label"):
        IssueRecord("org/repo", "2024-01-01", "support", "Title", "Body")


def test_compose_text_handles_empty_body() -> None:
    issue = IssueRecord("org/repo", "2024-01-01", "bug", "  Broken   build ", "")
    assert compose_text(issue) == "Broken build"
    assert compose_text(issue, fields=("title",)) == "Broken build"


def test_compose_text_rejects_unknown_field(sample_records: list[IssueRecord]) -> None:
    with pytest.raises(ValueError, match="unsupported text fields"):
        compose_text(sample_records[0], fields=("comments",))
