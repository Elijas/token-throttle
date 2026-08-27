"""Redis-safe backend fixtures for the multi-process scenarios."""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests._redis_guard import ensure_flush_allowed
from tests.multiprocess._harness import BackendSpec, scaled

# These tests flush their logical database, so they refuse to run against DB 0
# (the default a bare URL selects). Any explicitly chosen non-default database
# is accepted, which lets parallel workers each take their own index instead of
# contending on one — sharing a flushing database across runs produces phantom
# failures that look like product bugs.
FORBIDDEN_REDIS_DB = 0


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--redis-url"))


@dataclass(slots=True)
class BackendCase:
    """Parent-side backend controls paired with the child-build specification."""

    spec: BackendSpec
    observer: object | None = None

    def marker_is_live(self, reservation_id: str) -> bool:
        if self.spec.kind == "sqlite":
            if self.spec.locator is None:
                raise AssertionError("SQLite backend case has no database path")
            read_only_uri = f"{Path(self.spec.locator).resolve().as_uri()}?mode=ro"
            with sqlite3.connect(read_only_uri, uri=True) as connection:
                row = connection.execute(
                    "SELECT expires_at FROM acquire_markers "
                    "WHERE key_prefix = ? AND reservation_id = ?",
                    (self.spec.key_prefix, reservation_id),
                ).fetchone()
            return row is not None and float(row[0]) > time.time()
        if self.spec.kind != "redis" or self.observer is None:
            raise NotImplementedError(
                f"marker inspection is not implemented for {self.spec.kind}"
            )
        from token_throttle._limiter_backends._redis._keys import (  # noqa: PLC0415
            redis_acquired_marker_key,
        )

        return bool(
            self.observer.exists(
                redis_acquired_marker_key(self.spec.key_prefix, reservation_id)
            )
        )

    def wait_until_marker_expires(self, reservation_id: str, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while self.marker_is_live(reservation_id):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"reservation marker {reservation_id!r} did not expire "
                    f"within {timeout:g}s"
                )
            time.sleep(min(scaled(0.05), remaining))


def _redis_case(redis_url: str) -> tuple[BackendCase, object]:
    sync_redis = pytest.importorskip("redis", reason="redis package not installed")
    redis_exceptions = pytest.importorskip(
        "redis.exceptions", reason="redis package not installed"
    )
    client = sync_redis.from_url(redis_url)
    try:
        client.ping()
    except redis_exceptions.RedisError as exc:
        client.close()
        pytest.skip(f"Redis unavailable at {redis_url}: {exc}")

    selected_db = int(client.connection_pool.connection_kwargs.get("db", 0))
    if selected_db == FORBIDDEN_REDIS_DB:
        client.close()
        pytest.fail(
            "multi-process tests flush their logical database, so they require "
            "a dedicated one: pass --redis-url redis://localhost:6379/13 "
            "(or any other non-default index)",
            pytrace=False,
        )

    ensure_flush_allowed(redis_url)
    client.flushdb()
    prefix = f"mp-{uuid.uuid4().hex}"
    return (
        BackendCase(
            spec=BackendSpec(
                kind="redis",
                locator=redis_url,
                key_prefix=prefix,
                options=(("sleep_interval", 0.02),),
            ),
            observer=client,
        ),
        client,
    )


def _sqlite_case(tmp_path: Path) -> BackendCase:
    return BackendCase(
        spec=BackendSpec(
            kind="sqlite",
            locator=str(tmp_path / "shared-state.sqlite3"),
            key_prefix=f"mp-{uuid.uuid4().hex}",
            options=(("sleep_interval", 0.02),),
        )
    )


@pytest.fixture
def sqlite_backend_case(tmp_path: Path) -> BackendCase:
    """Yield a SQLite-only backend case without evaluating Redis fixtures."""
    return _sqlite_case(tmp_path)


@pytest.fixture(
    params=[
        pytest.param("redis", marks=pytest.mark.redis, id="redis"),
        pytest.param("sqlite", id="sqlite"),
    ]
)
def shared_backend_case(
    request: pytest.FixtureRequest,
    redis_url: str,
    tmp_path: Path,
):
    """Yield each cross-process-capable backend without coupling their setup."""
    if request.param == "sqlite":
        yield _sqlite_case(tmp_path)
        return
    if request.param != "redis":
        raise ValueError(f"Unknown shared backend: {request.param}")
    case, client = _redis_case(redis_url)
    try:
        yield case
    finally:
        client.flushdb()
        client.close()


@pytest.fixture(
    params=[
        pytest.param("redis", marks=pytest.mark.redis, id="redis"),
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "memory",
            id="memory",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "MemoryBackend state is process-local; a fresh interpreter "
                    "cannot observe prior consumption"
                ),
            ),
        ),
    ]
)
def fresh_interpreter_backend_case(
    request: pytest.FixtureRequest,
    redis_url: str,
    tmp_path: Path,
):
    """Include memory as a strict expected-failure control for the process cliff."""
    if request.param == "memory":
        yield BackendCase(
            spec=BackendSpec(kind="memory", locator=None, key_prefix="unused")
        )
        return
    if request.param == "sqlite":
        yield _sqlite_case(tmp_path)
        return
    if request.param != "redis":
        raise ValueError(f"Unknown fresh-interpreter backend: {request.param}")
    case, client = _redis_case(redis_url)
    try:
        yield case
    finally:
        client.flushdb()
        client.close()
