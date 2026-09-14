from __future__ import annotations

import asyncio
import threading

import pytest

from token_throttle import PerModelConfig, Quota, SqliteBackendBuilder, UsageQuotas


@pytest.mark.parametrize("retry_method", ["aclose", "close"])
async def test_cancelled_builder_close_retains_remaining_backends_for_retry(
    tmp_path, monkeypatch, retry_method
):
    builder = SqliteBackendBuilder(tmp_path / "close.db", key_prefix="close")
    backends = [
        await builder.build_async(
            PerModelConfig(
                model_family=family,
                quotas=UsageQuotas(
                    [Quota(metric="requests", limit=10, per_seconds=10)]
                ),
            )
        )
        for family in ("first", "second", "third")
    ]
    entered, release = threading.Event(), threading.Event()
    real_close = backends[0]._engine.close

    def gated_close():
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test did not release engine close")
        real_close()

    monkeypatch.setattr(backends[0]._engine, "close", gated_close)
    closing = asyncio.create_task(builder.aclose())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        closing.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        if retry_method == "aclose":
            await builder.aclose()
            await builder.aclose()
        else:
            builder.close()
            builder.close()
        assert not builder._backends
        for backend in backends:
            assert backend._engine._closed
            assert all(not thread.is_alive() for thread in backend._executor._threads)
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        # Explicit handles ensure the regression itself never leaks on failure.
        for backend in backends:
            await backend.aclose()
        await builder.aclose()
