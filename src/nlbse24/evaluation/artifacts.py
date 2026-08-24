"""Versioned, atomic result artifacts consumed by P5."""

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from nlbse24.constants import UPSTREAM_COMMIT
from nlbse24.domain import IssueRecord

SCHEMA_VERSION = "1.0"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "__", value).strip("_.-")
    if not slug:
        raise ValueError(f"cannot create a slug from {value!r}")
    return slug.lower()


def fingerprint_records(records: Sequence[IssueRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = json.dumps(record.as_dict(), sort_keys=True, ensure_ascii=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def environment_metadata() -> dict[str, object]:
    packages = {}
    for package in ("joblib", "numpy", "pandas", "psutil", "scikit-learn"):
        packages[package] = importlib.metadata.version(package)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_run_artifact(
    *,
    protocol: str,
    fold: str,
    repository: str,
    seed: int,
    model_name: str,
    model_config: dict[str, Any],
    train_records: Sequence[IssueRecord],
    test_records: Sequence[IssueRecord],
    classes: Sequence[str],
    metrics: dict[str, Any],
    resources: dict[str, Any],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    scores: np.ndarray | None,
) -> dict[str, Any]:
    return json_ready(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": str(uuid.uuid4()),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "protocol": protocol,
            "fold": fold,
            "repository": repository,
            "seed": seed,
            "model": {"name": model_name, "config": model_config},
            "data": {
                "upstream_commit": UPSTREAM_COMMIT,
                "train_size": len(train_records),
                "test_size": len(test_records),
                "train_fingerprint": fingerprint_records(train_records),
                "test_fingerprint": fingerprint_records(test_records),
            },
            "classes": list(classes),
            "metrics": metrics,
            "resources": resources,
            "predictions": {
                "y_true": list(y_true),
                "y_pred": list(y_pred),
                "scores": scores,
            },
            "environment": environment_metadata(),
        }
    )


class ResultWriter:
    def __init__(self, root: str | Path, overwrite: bool = False) -> None:
        self.root = Path(root)
        self.overwrite = overwrite

    def artifact_path(self, artifact: dict[str, Any]) -> Path:
        return (
            self.root
            / slugify(str(artifact["protocol"]))
            / slugify(str(artifact["model"]["name"]))
            / slugify(str(artifact["repository"]))
            / str(artifact["seed"])
            / f"{slugify(str(artifact['fold']))}.json"
        )

    def write(self, artifact: dict[str, Any]) -> Path:
        if artifact.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing result schema version")
        path = self.artifact_path(artifact)
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"result already exists: {path}; pass overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def write_summary(
        self, protocol: str, model_name: str, seed: int, summary: dict[str, Any]
    ) -> Path:
        path = (
            self.root
            / slugify(protocol)
            / slugify(model_name)
            / f"summary-seed-{seed}.json"
        )
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"summary already exists: {path}; pass overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "model": model_name,
            "seed": seed,
            **summary,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path
