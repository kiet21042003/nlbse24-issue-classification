"""Reproducible data-quality and cross-split leakage audit."""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from nlbse24.domain import IssueRecord
from nlbse24.text import compose_text, normalize_whitespace

RecordKey = tuple[str, ...]


def _normalized(value: str) -> str:
    return normalize_whitespace(value).casefold()


def _exact_key(record: IssueRecord) -> RecordKey:
    return record.repo, _normalized(record.title), _normalized(record.body)


def _title_key(record: IssueRecord) -> RecordKey:
    return record.repo, _normalized(record.title)


def _duplicate_rows(
    records: Sequence[IssueRecord], key_function: Callable[[IssueRecord], RecordKey]
) -> int:
    counts = Counter(key_function(record) for record in records)
    return sum(count - 1 for count in counts.values() if count > 1)


def _conflicting_keys(
    records: Sequence[IssueRecord], key_function: Callable[[IssueRecord], RecordKey]
) -> int:
    labels_by_key: dict[RecordKey, set[str]] = defaultdict(set)
    for record in records:
        labels_by_key[key_function(record)].add(record.label)
    return sum(len(labels) > 1 for labels in labels_by_key.values())


def _length_statistics(records: Sequence[IssueRecord]) -> dict[str, float]:
    lengths = np.asarray([len(compose_text(record).split()) for record in records], dtype=float)
    if not len(lengths):
        return {"mean_words": 0.0, "median_words": 0.0, "p95_words": 0.0, "max_words": 0.0}
    return {
        "mean_words": float(np.mean(lengths)),
        "median_words": float(np.median(lengths)),
        "p95_words": float(np.percentile(lengths, 95)),
        "max_words": float(np.max(lengths)),
    }


def _split_audit(records: Sequence[IssueRecord]) -> dict[str, Any]:
    counts = Counter((record.repo, record.label) for record in records)
    repositories = sorted({record.repo for record in records})
    labels = sorted({record.label for record in records})
    created_at = sorted(record.created_at for record in records)
    return {
        "rows": len(records),
        "repositories": repositories,
        "labels": labels,
        "counts": {
            repo: {label: counts[(repo, label)] for label in labels} for repo in repositories
        },
        "missing_body": sum(not record.body.strip() for record in records),
        "duplicate_exact_rows": _duplicate_rows(records, _exact_key),
        "duplicate_titles_within_repository": _duplicate_rows(records, _title_key),
        "conflicting_exact_text_labels": _conflicting_keys(records, _exact_key),
        "created_at_min": created_at[0] if created_at else None,
        "created_at_max": created_at[-1] if created_at else None,
        "text_length": _length_statistics(records),
    }


def audit_dataset_splits(
    train_records: Sequence[IssueRecord], test_records: Sequence[IssueRecord]
) -> dict[str, Any]:
    """Audit both official splits and detect exact or title-level overlap."""

    train_exact = {_exact_key(record) for record in train_records}
    test_exact = {_exact_key(record) for record in test_records}
    train_titles = {_title_key(record) for record in train_records}
    test_titles = {_title_key(record) for record in test_records}
    train_by_exact: dict[RecordKey, list[IssueRecord]] = defaultdict(list)
    test_by_exact: dict[RecordKey, list[IssueRecord]] = defaultdict(list)
    for record in train_records:
        train_by_exact[_exact_key(record)].append(record)
    for record in test_records:
        test_by_exact[_exact_key(record)].append(record)
    shared_exact = sorted(train_exact.intersection(test_exact))
    examples = [
        {
            "repository": key[0],
            "title": train_by_exact[key][0].title,
            "train_labels": sorted({record.label for record in train_by_exact[key]}),
            "test_labels": sorted({record.label for record in test_by_exact[key]}),
        }
        for key in shared_exact[:10]
    ]
    combined = [*train_records, *test_records]
    return {
        "train": _split_audit(train_records),
        "test": _split_audit(test_records),
        "cross_split": {
            "shared_exact_texts": len(shared_exact),
            "shared_exact_examples": examples,
            "shared_titles_within_repository": len(train_titles.intersection(test_titles)),
            "conflicting_exact_text_labels": _conflicting_keys(combined, _exact_key),
            "conflicting_title_labels": _conflicting_keys(combined, _title_key),
        },
    }
