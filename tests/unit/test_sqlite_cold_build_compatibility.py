from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from token_throttle import (
    MemoryBackendBuilder,
    PerModelConfig,
    Quota,
    RateLimiter,
    SqliteBackendBuilder,
    UsageQuotas,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="cold-build-compatibility",
        quotas=UsageQuotas([Quota(metric="requests", limit=10, per_seconds=60)]),
    )


async def test_cold_build_compatibility_honors_subclass_override(tmp_path: Path):
    calls: list[int] = []

    class CustomBuilder(SqliteBackendBuilder):
        def build(self, cfg, *, callbacks=None):
            calls.append(threading.get_ident())
            return super().build(cfg, callbacks=callbacks)

    builder = CustomBuilder(tmp_path / "subclass.sqlite3", key_prefix="compat")
    async with RateLimiter(_config(), backend=builder) as limiter:
        reservation = await limiter.acquire_capacity({"requests": 1}, "model")
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert calls == [threading.get_ident()]


async def test_cold_build_compatibility_honors_instance_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    builder = SqliteBackendBuilder(tmp_path / "instance.sqlite3", key_prefix="compat")
    original_build = builder.build
    calls: list[int] = []

    def custom_build(cfg, *, callbacks=None):
        calls.append(threading.get_ident())
        return original_build(cfg, callbacks=callbacks)

    monkeypatch.setattr(builder, "build", custom_build)
    async with RateLimiter(_config(), backend=builder) as limiter:
        reservation = await limiter.acquire_capacity({"requests": 1}, "model")
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert calls == [threading.get_ident()]


async def test_cold_build_compatibility_ignores_unrelated_custom_async_method():
    calls: list[str] = []

    class CustomBuilder(MemoryBackendBuilder):
        def build(self, cfg, *, callbacks=None):
            calls.append("build")
            return super().build(cfg, callbacks=callbacks)

        async def build_async(self, cfg):
            calls.append("unrelated")
            raise AssertionError("unrelated build_async must not be called")

    async with RateLimiter(_config(), backend=CustomBuilder()) as limiter:
        reservation = await limiter.acquire_capacity({"requests": 1}, "model")
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert calls == ["build"]


async def test_cold_build_compatibility_ignores_unrelated_sqlite_async_method(
    tmp_path: Path,
):
    class CustomBuilder(SqliteBackendBuilder):
        async def build_async(self, cfg):
            raise AssertionError("unrelated build_async must not be called")

    builder = CustomBuilder(tmp_path / "unrelated.sqlite3", key_prefix="compat")
    async with RateLimiter(_config(), backend=builder) as limiter:
        reservation = await limiter.acquire_capacity({"requests": 1}, "model")
        await limiter.refund_capacity({"requests": 0}, reservation)


async def test_cold_build_custom_builder_deliberately_opts_into_async_dispatch():
    calls: list[tuple[PerModelConfig, object, float | None]] = []

    class CustomBuilder(MemoryBackendBuilder):
        def build(self, cfg, *, callbacks=None):
            raise AssertionError("explicit async opt-in should take precedence")

        async def build_async(self, cfg):
            raise AssertionError("unrelated build_async must not be called")

        async def __token_throttle_async_build__(
            self, cfg, *, callbacks=None, timeout=None
        ):
            calls.append((cfg, callbacks, timeout))
            return super().build(cfg, callbacks=callbacks)

    async with RateLimiter(_config(), backend=CustomBuilder()) as limiter:
        reservation = await limiter.acquire_capacity(
            {"requests": 1}, "model", timeout=1
        )
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert len(calls) == 1
        cfg, callbacks, timeout = calls[0]
        assert cfg.get_model_family() == "cold-build-compatibility"
        assert [quota.limit for quota in cfg.quotas] == [10]
        assert callbacks is limiter._backend_callbacks
        assert timeout == 1


async def test_cold_build_sqlite_subclass_deliberately_opts_into_async_dispatch(
    tmp_path: Path,
):
    calls: list[float | None] = []

    class CustomBuilder(SqliteBackendBuilder):
        def build(self, cfg, *, callbacks=None):
            raise AssertionError("explicit async opt-in should take precedence")

        async def __token_throttle_async_build__(
            self, cfg, *, callbacks=None, timeout=None
        ):
            calls.append(timeout)
            return await SqliteBackendBuilder.build_async(
                self, cfg, callbacks=callbacks, timeout=timeout
            )

    builder = CustomBuilder(tmp_path / "opt-in.sqlite3", key_prefix="compat")
    async with RateLimiter(_config(), backend=builder) as limiter:
        reservation = await limiter.acquire_capacity(
            {"requests": 1}, "model", timeout=0
        )
        await limiter.refund_capacity({"requests": 0}, reservation)
        assert calls == [0]
