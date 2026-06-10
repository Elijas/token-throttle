r"""
CLI entry point for the token-throttle benchmark harness.

Examples::

    uv run python -m benchmarks.run                       # all memory scenarios
    uv run python -m benchmarks.run --scenario memory     # memory only
    uv run python -m benchmarks.run --scenario redis \
        --redis-url redis://localhost:6379/13             # Redis scenarios
    uv run python -m benchmarks.run -n 2000 -c 8          # iterations / concurrency
    uv run python -m benchmarks.run --json results.json   # machine-readable copy

Absolute numbers are not authoritative: they depend on the machine, system load,
Python build, and (for Redis) network/locality. The harness reports what it
measured on the host it ran on; see ``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks._workloads import (
    RunContext,
    _RedisNonEmptyError,
    _RedisUnavailableError,
    scenario_names,
    select_scenarios,
)

if TYPE_CHECKING:
    from benchmarks._stats import ScenarioResult

_DEFAULT_ITERATIONS = 2000
_DEFAULT_WARMUP_FRACTION = 0.1
_MIN_WARMUP = 5
_DEFAULT_CONCURRENCY = 4

_COLUMNS = (
    ("scenario", 34, "<"),
    ("conc", 5, ">"),
    ("iters", 7, ">"),
    ("p50_us", 11, ">"),
    ("p90_us", 11, ">"),
    ("p99_us", 11, ">"),
    ("mean_us", 11, ">"),
    ("ops/sec", 12, ">"),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks.run",
        description="Characterize token-throttle acquire-path overhead.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        metavar="NAME_OR_SUBSTRING",
        help=(
            "Scenario to run (repeatable). Matches an exact name or a substring "
            "(e.g. 'memory', 'redis', 'contended'). Default: all non-Redis "
            "scenarios. Available: " + ", ".join(scenario_names())
        ),
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=_DEFAULT_ITERATIONS,
        help=f"Timed acquire+refund operations per scenario (default {_DEFAULT_ITERATIONS}).",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help=f"Worker count for contended scenarios (default {_DEFAULT_CONCURRENCY}).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=(
            "Discarded warmup operations per worker before timing "
            f"(default max({_MIN_WARMUP}, {_DEFAULT_WARMUP_FRACTION} x iterations))."
        ),
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help=(
            "Redis URL for Redis scenarios, e.g. redis://localhost:6379/13. "
            "Redis scenarios are skipped if omitted or if Redis is unreachable. "
            "The target database must be EMPTY; the harness never flushes and "
            "cleans up only the keys it creates under a unique prefix."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        metavar="PATH",
        help="Also write the full results (with environment metadata) as JSON to PATH.",
    )
    return parser


def _environment_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _print_header(env: dict[str, str], iterations: int, concurrency: int) -> None:
    print("token-throttle acquire-path benchmark")
    print("=" * 72)
    print(f"python      : {env['python_implementation']} {env['python_version']}")
    print(f"platform    : {env['platform']}")
    print(f"processor   : {env['processor']}")
    print(f"timestamp   : {env['timestamp_utc']}")
    print(f"iterations  : {iterations}    contended concurrency: {concurrency}")
    print(
        "CAVEAT      : absolute numbers depend on machine, load, Python build, "
        "and Redis locality."
    )
    print(
        "              Use the baseline_noop row to subtract harness overhead. "
        "Treat as relative."
    )
    print("=" * 72)


def _format_cell(value: object, width: int, align: str) -> str:
    text = f"{value:,.2f}" if isinstance(value, float) else str(value)
    return f"{text:{align}{width}}"


def _print_table(results: list[ScenarioResult]) -> None:
    header = "  ".join(
        _format_cell(name, width, align) for name, width, align in _COLUMNS
    )
    print(header)
    print("-" * len(header))
    for result in results:
        row = (
            result.name,
            result.concurrency,
            result.iterations,
            result.p50_us,
            result.p90_us,
            result.p99_us,
            result.mean_us,
            result.ops_per_second,
        )
        print(
            "  ".join(
                _format_cell(value, width, align)
                for value, (_, width, align) in zip(row, _COLUMNS, strict=True)
            )
        )


def _resolve_warmup(iterations: int, warmup_arg: int | None) -> int:
    if warmup_arg is not None:
        if warmup_arg < 0:
            raise ValueError("--warmup must be >= 0")
        return warmup_arg
    return max(_MIN_WARMUP, int(iterations * _DEFAULT_WARMUP_FRACTION))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # The harness constructs and closes many limiters/backends; their normal
    # close-time INFO/WARNING logging would flood the table. Silence it so the
    # only output is the benchmark report. This does not affect the measured
    # paths -- it only changes log verbosity.
    logging.getLogger("token_throttle").setLevel(logging.ERROR)

    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")

    warmup = _resolve_warmup(args.iterations, args.warmup)
    ctx = RunContext(
        iterations=args.iterations,
        warmup=warmup,
        concurrency=args.concurrency,
        redis_url=args.redis_url,
    )

    try:
        specs = select_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    # Default (no explicit selection) skips Redis scenarios so the harness runs
    # with zero external services. Redis scenarios run when explicitly selected.
    if args.scenarios is None:
        specs = [spec for spec in specs if not spec.needs_redis]

    env = _environment_metadata()
    _print_header(env, args.iterations, args.concurrency)

    results: list[ScenarioResult] = []
    skipped: list[tuple[str, str]] = []
    for spec in specs:
        if spec.needs_redis and ctx.redis_url is None:
            skipped.append((spec.name, "no --redis-url provided"))
            continue
        try:
            results.append(spec.run(ctx))
        except _RedisUnavailableError as exc:
            skipped.append((spec.name, f"Redis unavailable: {exc}"))
        except _RedisNonEmptyError as exc:
            skipped.append((spec.name, str(exc)))

    if results:
        _print_table(results)
    else:
        print("(no scenarios produced results)")

    if skipped:
        print()
        print("Skipped scenarios:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if args.json_path is not None:
        payload = {
            "environment": env,
            "config": {
                "iterations": args.iterations,
                "warmup": warmup,
                "concurrency": args.concurrency,
                "redis_url": args.redis_url,
            },
            "results": [result.as_dict() for result in results],
            "skipped": [{"scenario": n, "reason": r} for n, r in skipped],
        }
        Path(args.json_path).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote JSON results to {args.json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
