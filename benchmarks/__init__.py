"""
Stdlib-only performance benchmark harness for token-throttle.

Characterizes acquire-path overhead for the memory and Redis backends across
sync/async APIs and uncontended/contended workloads. No third-party benchmark
dependency: timing uses ``time.perf_counter`` and statistics are computed in
pure Python.

Run with ``uv run python -m benchmarks.run``; see ``benchmarks/README.md`` for
how to interpret the output and the strong caveat that absolute numbers are
machine- and Redis-locality-dependent.
"""
