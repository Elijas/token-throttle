"""Redis-safe backend fixtures for the multi-process scenarios."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import pytest

from tests._redis_guard import ensure_flush_allowed
from tests.multiprocess._harness import BackendSpec, scaled

REQUIRED_REDIS_DB = 13


@pytest.fixture
def redis_url(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--redis-url"))


@dataclass(slots=True)
class BackendCase:
    """Parent-side backend controls paired with the child-build specification."""

    spec: BackendSpec
    observer: object | None = None

    def marker_exists(self, reservation_id: str) -> bool:
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
        while self.marker_exists(reservation_id):
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
    if selected_db != REQUIRED_REDIS_DB:
        client.close()
        pytest.fail(
            "multi-process tests require a dedicated Redis logical database: "
            f"pass --redis-url redis://localhost:6379/{REQUIRED_REDIS_DB} "
            f"(configured DB is {selected_db})",
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


@pytest.fixture(params=[pytest.param("redis", marks=pytest.mark.redis)])
def shared_backend_case(request: pytest.FixtureRequest, redis_url: str):
    """Yield shared backends; add SQLite to this parameter list when it lands."""
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
        pytest.param("redis", marks=pytest.mark.redis),
        pytest.param(
            "memory",
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
def fresh_interpreter_backend_case(request: pytest.FixtureRequest, redis_url: str):
    """Include memory as a strict expected-failure control for the process cliff."""
    if request.param == "memory":
        yield BackendCase(
            spec=BackendSpec(kind="memory", locator=None, key_prefix="unused")
        )
        return
    case, client = _redis_case(redis_url)
    try:
        yield case
    finally:
        client.flushdb()
        client.close()
