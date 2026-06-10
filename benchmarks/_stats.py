"""
Latency-sample statistics and the result record shared across scenarios.

Pure stdlib. Latencies are stored in seconds (the unit of ``time.perf_counter``)
and rendered as microseconds in the table because acquire-path operations are
sub-millisecond on the memory backend.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScenarioResult:
    """
    Aggregated timing for one benchmark scenario.

    ``latencies_seconds`` are per-operation wall-clock samples. For contended
    scenarios each worker's operations all contribute samples, so the latency
    percentiles describe per-operation latency under contention while
    ``ops_per_second`` describes aggregate throughput across all workers.
    """

    name: str
    backend: str
    api: str
    concurrency: int
    iterations: int
    wall_seconds: float
    p50_us: float
    p90_us: float
    p99_us: float
    mean_us: float
    min_us: float
    max_us: float
    ops_per_second: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(sorted_samples: list[float], fraction: float) -> float:
    """
    Nearest-rank percentile over an already-sorted, non-empty list.

    Nearest-rank (rather than interpolation) is used so a reported percentile is
    always an actually-observed sample, which is the honest reading for a small,
    noisy benchmark set.
    """
    if not sorted_samples:
        raise ValueError("cannot compute a percentile of zero samples")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    rank = math.ceil(fraction * len(sorted_samples))
    index = max(0, min(rank - 1, len(sorted_samples) - 1))
    return sorted_samples[index]


def summarize(  # noqa: PLR0913 - keyword-only result fields kept explicit for readability
    *,
    name: str,
    backend: str,
    api: str,
    concurrency: int,
    latencies_seconds: list[float],
    wall_seconds: float,
) -> ScenarioResult:
    """
    Reduce raw per-operation latency samples to a ``ScenarioResult``.

    ``wall_seconds`` is the measured wall-clock duration of the timed phase and
    is the basis for ``ops_per_second`` (total operations / wall time), so it
    reflects real aggregate throughput including contention overlap rather than
    the sum of isolated per-op latencies.
    """
    if not latencies_seconds:
        raise ValueError(f"scenario {name!r} produced zero latency samples")
    if wall_seconds <= 0.0:
        raise ValueError(
            f"scenario {name!r} reported non-positive wall time {wall_seconds}"
        )

    ordered = sorted(latencies_seconds)
    iterations = len(ordered)
    to_us = 1_000_000.0
    return ScenarioResult(
        name=name,
        backend=backend,
        api=api,
        concurrency=concurrency,
        iterations=iterations,
        wall_seconds=wall_seconds,
        p50_us=_percentile(ordered, 0.50) * to_us,
        p90_us=_percentile(ordered, 0.90) * to_us,
        p99_us=_percentile(ordered, 0.99) * to_us,
        mean_us=(sum(ordered) / iterations) * to_us,
        min_us=ordered[0] * to_us,
        max_us=ordered[-1] * to_us,
        ops_per_second=iterations / wall_seconds,
    )
