"""Đọc và tổng hợp kết quả của các lần chạy thí nghiệm.

Module này xử lý các artifact JSON được tạo ra sau khi đánh giá mô hình phân loại.
Mỗi artifact chứa thông tin về lần chạy, giao thức đánh giá, fold/repository,
seed, mô hình, tập dữ liệu, chỉ số đánh giá, tài nguyên sử dụng, dự đoán và môi
trường thực thi.

Các chức năng chính:

* Kiểm tra artifact có đủ trường bắt buộc, đúng phiên bản schema và giao thức
  được hỗ trợ; đồng thời kiểm tra cấu trúc mô hình và dữ liệu dự đoán cơ bản.
* Tìm toàn bộ artifact JSON trong một thư mục kết quả, bỏ qua các tệp tổng hợp
  có tên bắt đầu bằng ``summary-seed-``.
* Đọc một hoặc nhiều artifact, với tùy chọn bật hoặc tắt bước kiểm tra dữ liệu.
* Chuyển cấu trúc JSON lồng nhau thành một hàng phẳng gồm metadata, accuracy,
  các chỉ số macro/weighted, chỉ số theo từng lớp và thông tin tài nguyên dùng
  khi huấn luyện hoặc suy luận.
* Gom các hàng kết quả thành ``pandas.DataFrame`` để thuận tiện phân tích, so
  sánh hoặc xuất báo cáo.

Luồng sử dụng thông thường là gọi ``results_dataframe(root)``. Hàm này sẽ lần
lượt tìm tệp, đọc, kiểm tra, làm phẳng và trả về một DataFrame tổng hợp.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from nlbse24.evaluation.artifacts import SCHEMA_VERSION

_REQUIRED_KEYS = {
    "schema_version",
    "run_id",
    "created_at_utc",
    "protocol",
    "fold",
    "repository",
    "seed",
    "model",
    "data",
    "classes",
    "metrics",
    "resources",
    "predictions",
    "environment",
}


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Perform lightweight validation without adding a new dependency."""

    missing = sorted(_REQUIRED_KEYS - artifact.keys())
    if missing:
        raise ValueError(f"result artifact is missing required keys: {missing}")

    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported result schema version: {artifact['schema_version']!r}"
        )

    if artifact["protocol"] not in {"cv", "loo", "official"}:
        raise ValueError(f"unsupported protocol: {artifact['protocol']!r}")

    model = artifact["model"]
    if not isinstance(model, dict) or not model.get("name"):
        raise ValueError("artifact.model.name must be a non-empty string")

    predictions = artifact["predictions"]
    if not isinstance(predictions, dict):
        raise ValueError("artifact.predictions must be an object")
    for key in ("y_true", "y_pred", "scores"):
        if key not in predictions:
            raise ValueError(f"artifact.predictions is missing {key!r}")

    if len(predictions["y_true"]) != len(predictions["y_pred"]):
        raise ValueError("artifact y_true and y_pred must have equal length")


def load_artifact(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load one run artifact from JSON."""

    path = Path(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if validate:
        validate_artifact(artifact)
    return artifact


def find_artifact_paths(root: str | Path) -> list[Path]:
    """Find run artifacts while excluding summary JSON files."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"results directory does not exist: {root}")

    paths: list[Path] = []
    for path in root.rglob("*.json"):
        if path.name.startswith("summary-seed-"):
            continue
        paths.append(path)
    return sorted(paths)


def load_artifacts(root: str | Path, *, validate: bool = True) -> list[dict[str, Any]]:
    """Load all run artifacts below a results directory."""

    return [load_artifact(path, validate=validate) for path in find_artifact_paths(root)]


def artifact_to_row(artifact: dict[str, Any]) -> dict[str, Any]:
    """Flatten one run artifact into a table-friendly row."""

    metrics = artifact["metrics"]
    resources = artifact.get("resources", {})
    fit = resources.get("fit", {})
    inference = resources.get("inference", {})

    row: dict[str, Any] = {
        "run_id": artifact["run_id"],
        "created_at_utc": artifact["created_at_utc"],
        "protocol": artifact["protocol"],
        "fold": artifact["fold"],
        "repository": artifact["repository"],
        "seed": artifact["seed"],
        "model": artifact["model"]["name"],
        "train_size": artifact["data"]["train_size"],
        "test_size": artifact["data"]["test_size"],
        "train_fingerprint": artifact["data"]["train_fingerprint"],
        "test_fingerprint": artifact["data"]["test_fingerprint"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_average"]["precision"],
        "macro_recall": metrics["macro_average"]["recall"],
        "macro_f1": metrics["macro_average"]["f1-score"],
        "weighted_precision": metrics["weighted_average"]["precision"],
        "weighted_recall": metrics["weighted_average"]["recall"],
        "weighted_f1": metrics["weighted_average"]["f1-score"],
    }

    for label, values in metrics["per_class"].items():
        row[f"{label}_precision"] = values["precision"]
        row[f"{label}_recall"] = values["recall"]
        row[f"{label}_f1"] = values["f1-score"]
        row[f"{label}_support"] = values["support"]

    _add_profile_columns(row, "fit", fit)
    _add_profile_columns(row, "inference", inference)
    return row


def _add_profile_columns(
    row: dict[str, Any], prefix: str, profile: dict[str, Any]
) -> None:
    for key, value in profile.items():
        if isinstance(value, str | int | float | bool) or value is None:
            row[f"{prefix}_{key}"] = value


def artifacts_to_dataframe(artifacts: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert run artifacts to one analysis DataFrame."""

    rows = [artifact_to_row(artifact) for artifact in artifacts]
    return pd.DataFrame(rows)


def results_dataframe(root: str | Path, *, validate: bool = True) -> pd.DataFrame:
    """Convenience function: discover, load, validate, and flatten results."""

    return artifacts_to_dataframe(load_artifacts(root, validate=validate))
