"""Ước lượng khoảng tin cậy cho macro-F1 bằng phương pháp bootstrap.

Module này đo mức độ không chắc chắn của macro-F1 từ nhãn thực tế và nhãn dự
đoán của một lần đánh giá mô hình phân loại. Thay vì chỉ trả về một giá trị F1,
module lấy mẫu lại các phần tử nhiều lần (có hoàn lại), tính macro-F1 trên từng
mẫu và dùng các phân vị của tập kết quả để tạo khoảng tin cậy.

Các chức năng chính:

* ``bootstrap_macro_f1`` nhận ``y_true`` và ``y_pred``, kiểm tra đầu vào, tính
  macro-F1 gốc rồi thực hiện số lần lấy mẫu được cấu hình bởi ``n_resamples``.
  Kết quả gồm ước lượng gốc, cận dưới/cận trên, mức tin cậy, số lần lấy mẫu và
  seed dùng để tái lập thí nghiệm.
* ``bootstrap_artifact_macro_f1`` là hàm tiện ích dành cho artifact kết quả;
  hàm lấy trực tiếp ``y_true`` và ``y_pred`` từ trường ``predictions`` rồi gọi
  quy trình bootstrap chung.
* ``_macro_f1`` sử dụng ``evaluate_predictions`` của dự án để cách sắp xếp lớp
  và xử lý phép chia cho không luôn thống nhất với các lần đánh giá mô hình.

Các mẫu bootstrap được tạo bằng NumPy với cùng kích thước dữ liệu ban đầu và
lấy chỉ mục có hoàn lại. Khoảng tin cậy sử dụng phương pháp percentile: với mức
tin cậy mặc định 95%, hai cận là phân vị 2,5% và 97,5% của các macro-F1 thu được.
Seed mặc định giúp kết quả có thể tái lập.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from nlbse24.evaluation.metrics import evaluate_predictions


def bootstrap_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for macro-F1.

    Resampling is performed over examples with replacement. The point estimate
    uses the project's shared ``evaluate_predictions`` implementation so class
    ordering and zero-division behavior remain consistent with all model runs.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("cannot bootstrap empty predictions")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    true = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    n_samples = len(true)
    rng = np.random.default_rng(seed)

    point_estimate = _macro_f1(true, pred)
    bootstrap_scores = np.empty(n_resamples, dtype=float)

    for index in range(n_resamples):
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        bootstrap_scores[index] = _macro_f1(
            true[sample_indices], pred[sample_indices]
        )

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(
        bootstrap_scores, [alpha / 2.0, 1.0 - alpha / 2.0]
    )

    return {
        "metric": "macro_f1",
        "point_estimate": float(point_estimate),
        "confidence_level": float(confidence_level),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def bootstrap_artifact_macro_f1(
    artifact: dict[str, Any],
    *,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute macro-F1 CI directly from one versioned run artifact."""

    predictions = artifact["predictions"]
    return bootstrap_macro_f1(
        predictions["y_true"],
        predictions["y_pred"],
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    metrics = evaluate_predictions(list(y_true), list(y_pred))
    return float(metrics["macro_average"]["f1-score"])
