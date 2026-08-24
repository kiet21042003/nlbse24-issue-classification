"""Lightweight timing and memory measurement for the cost/performance RQ."""

import os
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import ParamSpec, TypeVar

import psutil

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    elapsed_seconds: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    python_peak_mb: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def profile_call(
    function: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> tuple[T, ResourceProfile]:
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()
    rss_after = process.memory_info().rss
    profile = ResourceProfile(
        elapsed_seconds=elapsed,
        rss_before_mb=rss_before / (1024**2),
        rss_after_mb=rss_after / (1024**2),
        rss_delta_mb=(rss_after - rss_before) / (1024**2),
        python_peak_mb=peak / (1024**2),
    )
    return result, profile
