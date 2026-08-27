"""
Run the CI gates locally, before pushing.

GitHub's hosted runners frequently queue for tens of minutes before a job
starts, so a push that fails CI costs far more wall-clock time than the job
itself suggests. This reproduces what CI checks, on this machine, so a red
run becomes surprising rather than routine.

Usage:
    uv run python -m devtools.preflight            # default tier
    uv run python -m devtools.preflight --quick    # fast feedback only
    uv run python -m devtools.preflight --full     # + the Python matrix

Tiers:
    quick    lint, format, type-check, unit, conformance, doc lints
    default  quick + integration, multiprocess, the dependency floor, and the
             newest redis client -- i.e. every job that gates a pull request
    full     default + the 3.12/3.13/3.14 matrix

What this cannot prove is listed at the end of every run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Candidate logical databases, probed for emptiness at run time. Each
# concurrent Redis-touching check needs its own: the suites flush around every
# test, so sharing one produces phantom failures that look like product bugs.
# DB 0 is never a candidate — it is the default a bare URL selects, and is the
# most likely to hold something real.
REDIS_DB_CANDIDATES = tuple(range(1, 16))
REDIS_HOST = os.environ.get("PREFLIGHT_REDIS_HOST", "redis://localhost:6379")

# Proven only by a real CI run; stated after every preflight so a green local
# result is never mistaken for a green matrix.
NOT_COVERED_LOCALLY = (
    "Windows (test-unit-platform windows-latest) — no Windows host here",
    "Linux specifics (container filesystem, service networking, umask)",
    "CodeQL analysis",
    "Codecov upload (needs the repository token and a real Actions run)",
)


@dataclass
class Check:
    name: str
    argv: list[str]
    tier: str = "quick"
    needs_redis: bool = False
    env: dict[str, str] = field(default_factory=dict)
    # When set, run in a throwaway environment so the dev venv is never mutated.
    isolated_python: str | None = None
    pre_argv: list[list[str]] = field(default_factory=list)
    # `uv sync --resolution ...` re-resolves and REWRITES uv.lock in the repo,
    # even when UV_PROJECT_ENVIRONMENT points the venv elsewhere — that only
    # redirects the environment, not the lockfile. Worse, a lock left recording
    # a different resolution mode makes every later `uv run` re-sync the real
    # environment. Checks flagged here get their own copy of the working tree,
    # and still run serially and last as a second line of defence.
    mutates_lockfile: bool = False


@dataclass
class Result:
    check: Check
    ok: bool
    seconds: float
    tail: str


def _redis_url(db: int) -> str:
    return f"{REDIS_HOST.rstrip('/')}/{db}"


def _checks(tier: str) -> list[Check]:
    """The local mirror of .github/workflows/ci.yml."""
    checks: list[Check] = [
        Check("lint", ["uv", "run", "ruff", "check", "."]),
        Check("format", ["uv", "run", "ruff", "format", "--check", "."]),
        Check("type-check", ["uv", "run", "mypy"]),
        Check("unit", ["uv", "run", "pytest", "tests/unit", "-q"], needs_redis=True),
        Check(
            "conformance",
            ["uv", "run", "pytest", "tests/conformance", "-q"],
            needs_redis=True,
        ),
        Check("doc-lints", ["uv", "run", "pytest", "tests/lint", "-q"]),
    ]
    if tier == "quick":
        return checks

    checks += [
        Check(
            "integration",
            ["uv", "run", "pytest", "tests/integration", "-q"],
            tier="default",
            needs_redis=True,
        ),
        Check(
            "multiprocess",
            ["uv", "run", "pytest", "tests/multiprocess", "-q"],
            tier="default",
            needs_redis=True,
            env={"TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE": "2"},
        ),
        # Mirrors test-min-deps: the lowest client the metadata permits. This is
        # the job that caught retry-replay tests depending on a version-specific
        # default, which the locked client hid completely.
        Check(
            "dependency-floor",
            [
                "uv",
                "run",
                "--no-sync",
                "pytest",
                "tests/unit",
                "tests/integration",
                "-m",
                "redis",
                "-q",
            ],
            tier="default",
            needs_redis=True,
            isolated_python="3.12",
            mutates_lockfile=True,
            pre_argv=[
                [
                    "uv",
                    "sync",
                    "--extra",
                    "redis",
                    "--group",
                    "dev",
                    "--resolution",
                    "lowest-direct",
                ],
            ],
        ),
        # Mirrors the scheduled drift canary: the newest client the uncapped
        # range admits, which uv.lock otherwise pins away from.
        Check(
            "newest-redis-client",
            [
                "uv",
                "run",
                "--no-sync",
                "pytest",
                "tests/unit",
                "tests/integration",
                "-m",
                "redis",
                "-q",
            ],
            tier="default",
            needs_redis=True,
            isolated_python="3.13",
            mutates_lockfile=True,
            pre_argv=[
                ["uv", "sync", "--all-extras", "--group", "dev"],
                ["uv", "pip", "install", "--upgrade", "redis"],
            ],
        ),
    ]
    if tier != "full":
        return checks

    for version in ("3.12", "3.13", "3.14"):
        checks.append(
            Check(
                f"unit-py{version}",
                ["uv", "run", "--no-sync", "pytest", "tests/unit", "-q"],
                tier="full",
                needs_redis=True,
                isolated_python=version,
                pre_argv=[["uv", "sync", "--all-extras", "--group", "dev"]],
            )
        )
    return checks


_COPY_SKIP = frozenset(
    {
        ".git",
        ".venv",
        ".worktrees",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        "dist",
        "htmlcov",
        "node_modules",
    }
)


def _copy_project(destination: Path) -> None:
    """Copy the working tree so a check can re-resolve without touching it."""
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(*_COPY_SKIP),
        symlinks=True,
    )


def _run(check: Check, db: int | None) -> Result:
    env = os.environ.copy()
    env.update(check.env)
    scratch: str | None = None
    cwd = REPO_ROOT
    if check.isolated_python is not None:
        # A separate environment keeps version-swapping checks from mutating
        # the dev venv -- an interrupted run must not leave a downgraded client
        # installed.
        scratch = tempfile.mkdtemp(prefix="tt-preflight-")
        env["UV_PROJECT_ENVIRONMENT"] = str(Path(scratch) / "venv")
        env["UV_PYTHON"] = check.isolated_python
    if check.mutates_lockfile:
        # UV_PROJECT_ENVIRONMENT redirects the virtualenv but NOT the lockfile:
        # `uv sync --resolution ...` still rewrites uv.lock in place, and a lock
        # left recording a different resolution mode makes every later `uv run`
        # re-sync the real environment. Give these checks their own copy of the
        # tree so the repository's lockfile is physically out of reach.
        if scratch is None:
            scratch = tempfile.mkdtemp(prefix="tt-preflight-")
            env["UV_PROJECT_ENVIRONMENT"] = str(Path(scratch) / "venv")
        cwd = Path(scratch) / "project"
        _copy_project(cwd)
        env["UV_PROJECT_ENVIRONMENT"] = str(Path(scratch) / "venv")

    argv = list(check.argv)
    if check.needs_redis and db is not None:
        argv += ["--redis-url", _redis_url(db)]

    started = time.monotonic()
    try:
        for pre in check.pre_argv:
            done = subprocess.run(
                pre, cwd=cwd, env=env, capture_output=True, text=True, check=False
            )
            if done.returncode != 0:
                return Result(
                    check,
                    ok=False,
                    seconds=time.monotonic() - started,
                    tail=f"setup failed: {' '.join(pre)}\n{done.stderr.strip()[-800:]}",
                )
        done = subprocess.run(
            argv, cwd=cwd, env=env, capture_output=True, text=True, check=False
        )
        output = (done.stdout + done.stderr).strip()
        return Result(
            check,
            ok=done.returncode == 0,
            seconds=time.monotonic() - started,
            tail="\n".join(output.splitlines()[-12:]),
        )
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _redis_reachable() -> bool:
    probe = shutil.which("redis-cli")
    if probe is None:
        return False
    done = subprocess.run([probe, "ping"], capture_output=True, text=True, check=False)
    return done.returncode == 0 and "PONG" in done.stdout


def _empty_redis_dbs(needed: int) -> list[int]:
    """
    Return `needed` logical databases that are currently empty.

    Probed rather than hardcoded: another project (or another preflight) may
    be holding keys in any given index, and handing a check a populated
    database means it either refuses to run or destroys someone's data.
    Returns fewer than requested if that many are not free; the caller decides.
    """
    probe = shutil.which("redis-cli")
    if probe is None:
        return []
    free: list[int] = []
    for db in REDIS_DB_CANDIDATES:
        done = subprocess.run(
            [probe, "-n", str(db), "dbsize"],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0 and done.stdout.strip() == "0":
            free.append(db)
        if len(free) == needed:
            break
    return free


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="fast feedback only")
    group.add_argument("--full", action="store_true", help="add the Python matrix")
    parser.add_argument(
        "--jobs", type=int, default=4, help="checks to run concurrently (default 4)"
    )
    args = parser.parse_args()

    tier = "quick" if args.quick else "full" if args.full else "default"
    checks = _checks(tier)

    if not _redis_reachable() and any(c.needs_redis for c in checks):
        print(
            "Redis is not answering at "
            f"{REDIS_HOST}. Start it before preflighting: the Redis-backed "
            "suites are where most CI surprises live.",
            file=sys.stderr,
        )
        return 2

    redis_checks = [c for c in checks if c.needs_redis]
    free_dbs = _empty_redis_dbs(len(redis_checks))
    if len(free_dbs) < len(redis_checks):
        print(
            f"Need {len(redis_checks)} empty Redis databases but found "
            f"{len(free_dbs)}. These suites flush the database they are given, "
            "so preflight will not reuse a populated one. Free some up, or run "
            "with --jobs 1 --quick for the checks that need no Redis.",
            file=sys.stderr,
        )
        return 2
    assigned: dict[str, int | None] = {c.name: None for c in checks}
    for check, db in zip(redis_checks, free_dbs, strict=False):
        assigned[check.name] = db

    concurrent = [c for c in checks if not c.mutates_lockfile]
    serial = [c for c in checks if c.mutates_lockfile]

    print(f"preflight [{tier}] — {len(checks)} checks, {args.jobs} at a time\n")
    started = time.monotonic()
    results: list[Result] = []

    def record(result: Result) -> None:
        results.append(result)
        mark = "PASS" if result.ok else "FAIL"
        print(f"  {mark}  {result.check.name:<22} {result.seconds:6.1f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run, c, assigned[c.name]) for c in concurrent]
        for future in as_completed(futures):
            record(future.result())

    # Re-resolution rewrites uv.lock in place, so these run one at a time, after
    # everything that reads the lockfile has finished, and the file is put back
    # exactly as found. A dev tool that silently upgrades pinned dependencies is
    # worse than no dev tool.
    if serial:
        lockfile = REPO_ROOT / "uv.lock"
        snapshot = lockfile.read_bytes() if lockfile.exists() else None
        try:
            for check in serial:
                record(_run(check, assigned[check.name]))
        finally:
            if snapshot is not None and lockfile.read_bytes() != snapshot:
                lockfile.write_bytes(snapshot)
                print("  ..    uv.lock changed unexpectedly; restored")

    failed = [r for r in results if not r.ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} passed "
        f"in {time.monotonic() - started:.0f}s wall clock"
    )
    for result in failed:
        print(f"\n----- {result.check.name} -----\n{result.tail}")
    print("\nNot covered locally — only a real CI run proves these:")
    for item in NOT_COVERED_LOCALLY:
        print(f"  - {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
