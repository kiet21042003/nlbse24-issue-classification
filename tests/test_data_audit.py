from nlbse24.data.audit import audit_dataset_splits
from nlbse24.domain import IssueRecord


def issue(label: str, title: str, body: str = "Body") -> IssueRecord:
    return IssueRecord("org/repo", "2024-01-01", label, title, body)


def test_audit_detects_cross_split_overlap_and_missing_body() -> None:
    train = [issue("bug", "Same title", "Same body"), issue("feature", "New option")]
    test = [issue("bug", " Same  title ", "same BODY"), issue("question", "Help", "")]
    report = audit_dataset_splits(train, test)
    assert report["cross_split"]["shared_exact_texts"] == 1
    assert report["cross_split"]["shared_exact_examples"][0]["train_labels"] == ["bug"]
    assert report["cross_split"]["shared_titles_within_repository"] == 1
    assert report["test"]["missing_body"] == 1


def test_audit_detects_conflicting_labels_for_same_title() -> None:
    train = [issue("bug", "Ambiguous", "First")]
    test = [issue("feature", "Ambiguous", "Second")]
    report = audit_dataset_splits(train, test)
    assert report["cross_split"]["conflicting_title_labels"] == 1
    assert report["cross_split"]["conflicting_exact_text_labels"] == 0
