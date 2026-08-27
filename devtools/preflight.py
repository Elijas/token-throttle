"""
Run the CI gates locally, before pushing.

GitHub's hosted runners frequently queue for tens of minutes before a job
starts, so a push that fails CI costs far more wall-clock time than the job
itself suggests. This reproduces what CI checks, on this machine, so a red
run becomes surprising rather than routine.

Usage:
    task preflight              # every job that gates a pull request
    task preflight -- --quick   # lint, types, unit, conformance, doc lints
    task preflight -- --full    # adds the 3.12/3.13/3.14 matrix

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
# DB 0 is never a candidate — it is what a bare URL selects and the most likely
# to hold something real.
REDIS_DB_CANDIDATES = tuple(range(1, 16))
REDIS_HOST = os.environ.get("PREFLIGHT_REDIS_HOST", "redis://localhost:6379")

# Substituted with the check's throwaway virtualenv path.
VENV = "{venv}"

# Proven only by a real CI run; printed after every preflight so a green local
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
    needs_redis: bool = False
    env: dict[str, str] = field(default_factory=dict)
    # Build a throwaway virtualenv on this Python first. Commands may reference
    # it as `{venv}`.
    venv_python: str | None = None
    pre_argv: list[list[str]] = field(default_factory=list)


@dataclass
class Result:
    check: Check
    ok: bool
    seconds: float
    tail: str


def _redis_url(db: int) -> str:
    return f"{REDIS_HOST.rstrip('/')}/{db}"


def _pytest(venv: str, *targets: str) -> list[str]:
    return [f"{venv}/bin/python", "-m", "pytest", *targets, "-q"]


def _pip_install(*args: str) -> list[str]:
    return ["uv", "pip", "install", "--python", VENV, "-q", *args]


def _checks(tier: str) -> list[Check]:
    """The local mirror of .github/workflows/ci.yml."""
    checks = [
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

    # The version-varying checks below install into a throwaway virtualenv with
    # `uv pip install` rather than `uv sync`. That is deliberate and measured:
    # `uv sync --resolution ...` rewrites the project's uv.lock even when
    # UV_PROJECT_ENVIRONMENT points the virtualenv elsewhere, and a lock left
    # recording a different resolution mode makes every later `uv run` re-sync
    # the real environment. `uv pip install` is pip-mode and neither reads nor
    # writes the lockfile. (`-e ".[redis]" --group dev` and
    # `-r pyproject.toml --extra redis --group dev` were measured to resolve
    # identically here — every direct dependency lands on its floor — and the
    # editable form also installs the project itself.)
    checks += [
        Check(
            "integration",
            ["uv", "run", "pytest", "tests/integration", "-q"],
            needs_redis=True,
        ),
        Check(
            "multiprocess",
            ["uv", "run", "pytest", "tests/multiprocess", "-q"],
            needs_redis=True,
            env={"TOKEN_THROTTLE_MULTIPROCESS_TIMING_SCALE": "2"},
        ),
        # Mirrors test-min-deps: the lowest client the metadata permits. This is
        # the check that caught retry-replay tests depending on a
        # version-specific default, which the locked client hid entirely.
        Check(
            "dependency-floor",
            _pytest(VENV, "tests/unit", "tests/integration", "-m", "redis"),
            needs_redis=True,
            venv_python="3.12",
            pre_argv=[
                _pip_install(
                    "--resolution", "lowest-direct", "-e", ".[redis]", "--group", "dev"
                ),
            ],
        ),
        # Mirrors the scheduled drift canary: the newest client the uncapped
        # range admits, which uv.lock otherwise pins away from.
        Check(
            "newest-redis-client",
            _pytest(VENV, "tests/unit", "tests/integration", "-m", "redis"),
            needs_redis=True,
            venv_python="3.13",
            pre_argv=[
                _pip_install("-e", ".[redis,tiktoken]", "--group", "dev"),
                _pip_install("--upgrade", "redis"),
            ],
        ),
    ]
    if tier != "full":
        return checks

    for version in ("3.12", "3.13", "3.14"):
        checks.append(
            Check(
                f"unit-py{version}",
                _pytest(VENV, "tests/unit"),
                needs_redis=True,
                venv_python=version,
                pre_argv=[_pip_install("-e", ".[redis,tiktoken]", "--group", "dev")],
            )
        )
    return checks


def _substitute(argv: list[str], venv: str | None) -> list[str]:
    if venv is None:
        return list(argv)
    return [part.replace(VENV, venv) for part in argv]


def _run(check: Check, db: int | None) -> Result:
    env = os.environ.copy()
    env.update(check.env)
    scratch: str | None = None
    venv: str | None = None
    started = time.monotonic()

    try:
        commands = list(check.pre_argv)
        if check.venv_python is not None:
            scratch = tempfile.mkdtemp(prefix="tt-preflight-")
            venv = str(Path(scratch) / "venv")
            commands.insert(
                0, ["uv", "venv", venv, "-q", "--python", check.venv_python]
            )

        for pre in commands:
            resolved = _substitute(pre, venv)
            done = subprocess.run(
                resolved,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                return Result(
                    check=check,
                    ok=False,
                    seconds=time.monotonic() - started,
                    tail=f"setup failed: {' '.join(resolved)}\n"
                    f"{(done.stdout + done.stderr).strip()[-800:]}",
                )

        argv = _substitute(check.argv, venv)
        if check.needs_redis and db is not None:
            argv += ["--redis-url", _redis_url(db)]
        done = subprocess.run(
            argv, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
        )
        output = (done.stdout + done.stderr).strip()
        return Result(
            check=check,
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
    hold keys in any given index, and handing a check a populated database
    means it either refuses to run or destroys someone's data.
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
            f"Redis is not answering at {REDIS_HOST}. Start it before "
            "preflighting: the Redis-backed suites are where most CI surprises "
            "live.",
            file=sys.stderr,
        )
        return 2

    redis_checks = [c for c in checks if c.needs_redis]
    free = _empty_redis_dbs(len(redis_checks))
    if len(free) < len(redis_checks):
        print(
            f"Need {len(redis_checks)} empty Redis databases, found {len(free)}. "
            "These suites flush the database they are handed, so preflight will "
            "not reuse a populated one. Free some up, or use --quick.",
            file=sys.stderr,
        )
        return 2
    assigned: dict[str, int | None] = {c.name: None for c in checks}
    for check, db in zip(redis_checks, free, strict=False):
        assigned[check.name] = db

    lockfile = REPO_ROOT / "uv.lock"
    lock_before = lockfile.read_bytes() if lockfile.exists() else None

    print(f"preflight [{tier}] — {len(checks)} checks, {args.jobs} at a time\n")
    started = time.monotonic()
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run, c, assigned[c.name]) for c in checks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            mark = "PASS" if result.ok else "FAIL"
            print(
                f"  {mark}  {result.check.name:<22} {result.seconds:6.1f}s", flush=True
            )

    # Nothing here should touch the lockfile. If that ever changes, say so
    # loudly rather than leaving a silently mutated dependency set behind.
    if lock_before is not None and lockfile.read_bytes() != lock_before:
        lockfile.write_bytes(lock_before)
        print("\n!! uv.lock was modified by a check and has been restored")

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
