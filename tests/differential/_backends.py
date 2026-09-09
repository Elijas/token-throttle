# ruff: noqa: TC003, PLR0913, PLR0915, PERF401, PLC0415
"""
Build the six built-in backends (memory/SQLite/Redis x async/sync) for one
config, each on isolated state, all reading the same ``FakeClock``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.differential._clock import FakeClock, bind_sqlite_engine_clock
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas

REDIS_URL_ENV = "TT_AUDIT_REDIS_URL"
DEFAULT_REDIS_URL = "redis://127.0.0.1:6399/13"
MAX_TTL_SECONDS = 2**31 - 1
ALL_KINDS: tuple[str, ...] = ("memory", "sqlite", "redis")
ALL_MODES: tuple[str, ...] = ("async", "sync")


def redis_url() -> str:
    return os.environ.get(REDIS_URL_ENV, DEFAULT_REDIS_URL)


def make_config(
    model_family: str,
    quotas: Sequence[tuple[str, int, float]],
) -> PerModelConfig:
    """``quotas`` = (metric, per_seconds, limit) triples; several windows per metric allowed."""
    return PerModelConfig(
        model_family=model_family,
        quotas=UsageQuotas(
            [
                Quota(metric=metric, per_seconds=per_seconds, limit=limit)
                for metric, per_seconds, limit in quotas
            ]
        ),
    )


@dataclass
class Harnessed:
    name: str
    kind: str
    is_async: bool
    backend: Any
    builder: Any
    cleanups: list[Callable[[], None]] = field(default_factory=list)

    def cleanup(self) -> None:
        errors: list[BaseException] = []
        for fn in reversed(self.cleanups):
            try:
                fn()
            except BaseException as exc:
                errors.append(exc)
        self.cleanups.clear()
        if errors:
            raise RuntimeError(f"{self.name}: cleanup failed: {errors!r}")


def _delete_prefix(client: Any, prefix: str) -> None:
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=f"{prefix}:*", count=500)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break


def _redis_reachable(url: str) -> bool:
    try:
        import redis as sync_redis
        from redis.exceptions import RedisError
    except ImportError:
        return False
    client = sync_redis.from_url(url)
    try:
        client.ping()
    except RedisError:
        return False
    finally:
        client.close()
    return True


def require_redis() -> str:
    url = redis_url()
    if not _redis_reachable(url):
        pytest.skip(f"Redis unavailable at {url} (set {REDIS_URL_ENV})")
    return url


def build_one(
    kind: str,
    mode: str,
    cfg: PerModelConfig,
    clock: FakeClock,
    *,
    loop: asyncio.AbstractEventLoop,
    tmp_path: Path,
    ttl_seconds: int = MAX_TTL_SECONDS,
    override_ttl_seconds: int | None = None,
    sleep_interval: float = 0.01,
    key_prefix: str | None = None,
    db_path: Path | None = None,
    max_reservation_lifetime_seconds: float | None = None,
) -> Harnessed:
    is_async = mode == "async"
    name = f"{kind}-{mode}"
    prefix = key_prefix or f"diff-{name}-{uuid.uuid4().hex}"
    if kind == "memory":
        if is_async:
            from token_throttle._limiter_backends._memory._backend import (
                MemoryBackendBuilder,
            )

            builder: Any = MemoryBackendBuilder(sleep_interval=sleep_interval)
        else:
            from token_throttle._limiter_backends._memory._sync_backend import (
                SyncMemoryBackendBuilder,
            )

            builder = SyncMemoryBackendBuilder(sleep_interval=sleep_interval)
        backend = builder.build(cfg)
        return Harnessed(name, kind, is_async, backend, builder)

    if kind == "sqlite":
        path = db_path or (tmp_path / f"{name}-{uuid.uuid4().hex}.sqlite3")
        kwargs: dict[str, Any] = {
            "key_prefix": prefix,
            "sleep_interval": sleep_interval,
            "bucket_ttl_seconds": ttl_seconds,
            "refund_dedup_ttl_seconds": ttl_seconds,
            "override_ttl_seconds": override_ttl_seconds or ttl_seconds,
            "max_reservation_lifetime_seconds": max_reservation_lifetime_seconds,
        }
        if is_async:
            from token_throttle._limiter_backends._sqlite._backend import (
                SqliteBackendBuilder,
            )

            builder = SqliteBackendBuilder(path, **kwargs)
        else:
            from token_throttle._limiter_backends._sqlite._sync_backend import (
                SyncSqliteBackendBuilder,
            )

            builder = SyncSqliteBackendBuilder(path, **kwargs)
        backend = builder.build(cfg)
        bind_sqlite_engine_clock(backend._engine, clock)
        harnessed = Harnessed(name, kind, is_async, backend, builder)
        if is_async:
            harnessed.cleanups.append(lambda: loop.run_until_complete(builder.aclose()))
        else:
            harnessed.cleanups.append(builder.close)
        return harnessed

    if kind == "redis":
        url = require_redis()
        import redis as sync_redis

        kwargs = {
            "key_prefix": prefix,
            "sleep_interval": sleep_interval,
            "bucket_ttl_seconds": ttl_seconds,
            "refund_dedup_ttl_seconds": ttl_seconds,
            "override_ttl_seconds": override_ttl_seconds or ttl_seconds,
        }
        if is_async:
            import redis.asyncio as async_redis

            from token_throttle._limiter_backends._redis._backend import (
                RedisBackendBuilder,
            )

            client = async_redis.from_url(url)
            builder = RedisBackendBuilder(client, **kwargs)
            backend = builder.build(cfg)
            harnessed = Harnessed(name, kind, is_async, backend, builder)
            harnessed.cleanups.append(lambda: loop.run_until_complete(builder.aclose()))
            harnessed.cleanups.append(lambda: loop.run_until_complete(client.aclose()))
        else:
            from token_throttle._limiter_backends._redis._sync_backend import (
                SyncRedisBackendBuilder,
            )

            client = sync_redis.from_url(url)
            builder = SyncRedisBackendBuilder(client, **kwargs)
            backend = builder.build(cfg)
            harnessed = Harnessed(name, kind, is_async, backend, builder)
            harnessed.cleanups.append(builder.close)
            harnessed.cleanups.append(client.close)

        def _purge() -> None:
            janitor = sync_redis.from_url(url)
            try:
                _delete_prefix(janitor, prefix)
            finally:
                janitor.close()

        harnessed.cleanups.append(_purge)
        return harnessed

    raise ValueError(f"unknown backend kind {kind!r}")


def build_all(
    cfg: PerModelConfig,
    clock: FakeClock,
    *,
    loop: asyncio.AbstractEventLoop,
    tmp_path: Path,
    kinds: Sequence[str] = ALL_KINDS,
    modes: Sequence[str] = ALL_MODES,
    **kwargs: Any,
) -> list[Harnessed]:
    built: list[Harnessed] = []
    try:
        for kind in kinds:
            for mode in modes:
                built.append(
                    build_one(
                        kind, mode, cfg, clock, loop=loop, tmp_path=tmp_path, **kwargs
                    )
                )
    except BaseException:
        for item in built:
            item.cleanup()
        raise
    return built
