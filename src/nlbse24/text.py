"""Deterministic issue-to-text transformations."""

import re
from collections.abc import Iterable

from nlbse24.domain import IssueRecord

SUPPORTED_TEXT_FIELDS = ("title", "body")


def normalize_whitespace(value: str) -> str:
    """Collapse whitespace without applying task-specific lexical cleanup."""

    return re.sub(r"\s+", " ", value).strip()


def compose_text(
    issue: IssueRecord,
    fields: tuple[str, ...] = SUPPORTED_TEXT_FIELDS,
    separator: str = "\n\n",
) -> str:
    """Compose model input from explicitly selected issue fields."""

    unknown = set(fields).difference(SUPPORTED_TEXT_FIELDS)
    if unknown:
        raise ValueError(f"unsupported text fields: {sorted(unknown)}")
    if not fields:
        raise ValueError("at least one text field is required")

    parts = [normalize_whitespace(getattr(issue, field)) for field in fields]
    return separator.join(part for part in parts if part)


def compose_texts(
    issues: Iterable[IssueRecord], fields: tuple[str, ...] = SUPPORTED_TEXT_FIELDS
) -> list[str]:
    return [compose_text(issue, fields=fields) for issue in issues]
