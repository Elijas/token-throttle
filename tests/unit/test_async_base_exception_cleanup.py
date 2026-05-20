"""Regression tests for async BaseException cleanup parity."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Any

import pytest
from frozendict import frozendict

from token_throttle._interfaces._callbacks import RateLimiterCallbacks
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._rate_limiter import RateLimiter

MODEL = "test-model"
MODEL_FAMILY = "test-family"
BUCKET_ID = ("tokens", 3600)


def _config(*, limit: float = 100.0) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [Quota(metric="tokens", limit=limit, per_seconds=BUCKET_ID[1])]
        ),
        model_family=MODEL_FAMILY,
    )


def _memory_capacity(backend) -> float:
    return backend._buckets[0].get_capacity(time.time()).amount


def _runtime_override(limiter: RateLimiter) -> float | None:
    return limiter._model_family_to_runtime_max_capacity.get(MODEL_FAMILY, {}).get(
        BUCKET_ID
    )


@pytest.mark.parametrize(
    "exception_type",
    [concurrent.futures.CancelledError, GeneratorExit],
)
async def test_memory_backend_refunds_post_consume_base_exception_callback(
    exception_type: type[BaseException],
) -> None:
    calls = 0

    async def callback(**_kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise exception_type

    backend = MemoryBackendBuilder().build(
        _config(),
        callbacks=RateLimiterCallbacks(on_capacity_consumed=callback),
    )

    await backend.await_for_capacity(frozendict({"tokens": 90.0}))
    capacity_before = _memory_capacity(backend)

    with pytest.raises(exception_type):
        await backend.await_for_capacity(frozendict({"tokens": 10.0}), timeout=0)

    assert _memory_capacity(backend) == pytest.approx(capacity_before, abs=1.0)


class _SystemExitBuilder:
    def build(self, _cfg: PerModelConfig, *, callbacks=None):
        _ = callbacks
        raise SystemExit("simulated backend setup interrupt")

    async def aclose(self) -> None:
        return None

    def close(self) -> None:
        return None


async def test_acquire_rolls_back_pending_state_on_outer_base_exception() -> None:
    limiter = RateLimiter(
        _config(),
        backend=_SystemExitBuilder(),
        close_drain_timeout_seconds=0.05,
    )

    with pytest.raises(SystemExit, match="simulated backend setup interrupt"):
        await limiter.acquire_capacity({"tokens": 1}, MODEL)

    assert limiter._pending_acquire_reservations == set()
    assert limiter._in_flight_reservation_ids == set()
    await asyncio.wait_for(limiter.aclose(), timeout=1.0)


@pytest.mark.parametrize("exception_type", [SystemExit, KeyboardInterrupt])
def test_public_limiter_close_drains_after_callback_systemexit(
    exception_type: type[BaseException],
) -> None:
    outcome: dict[str, Any] = {}

    async def scenario() -> None:
        async def callback(**_kwargs) -> None:
            raise exception_type("simulated callback interrupt")

        limiter = RateLimiter(
            _config(),
            backend=MemoryBackendBuilder(),
            callbacks=RateLimiterCallbacks(on_capacity_consumed=callback),
            close_drain_timeout_seconds=0.05,
        )

        try:
            await limiter.acquire_capacity({"tokens": 1}, MODEL)
        except BaseException as exc:
            outcome["public_exception_type"] = type(exc)
        else:  # pragma: no cover - the callback must interrupt the acquire.
            outcome["public_exception_type"] = None

        outcome["pending_acquires"] = set(limiter._pending_acquire_reservations)
        outcome["in_flight_reservations"] = limiter.snapshot_state()[
            "in_flight_reservations"
        ]
        try:
            await asyncio.wait_for(limiter.aclose(), timeout=1.0)
        except BaseException as exc:
            outcome["close_exception"] = exc
        else:
            outcome["close_exception"] = None

    with pytest.raises(exception_type, match="simulated callback interrupt"):
        asyncio.run(scenario())

    # CPython may deliver SystemExit/KeyboardInterrupt to the awaiting coroutine
    # directly or as CancelledError before re-raising the original at loop level.
    assert outcome["public_exception_type"] in {
        exception_type,
        asyncio.CancelledError,
    }
    assert outcome["pending_acquires"] == set()
    assert outcome["in_flight_reservations"] == 0
    assert outcome["close_exception"] is None


async def test_set_max_capacity_reconciles_post_write_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
    await limiter.acquire_capacity({"tokens": 1}, MODEL)
    backend = limiter._model_family_to_backend[MODEL_FAMILY]

    original_shield = asyncio.shield

    async def shield_then_raise(awaitable: Any) -> Any:
        await original_shield(awaitable)
        raise GeneratorExit("simulated post-write interrupt")

    monkeypatch.setattr(asyncio, "shield", shield_then_raise)

    with pytest.raises(GeneratorExit, match="simulated post-write interrupt"):
        await limiter.set_max_capacity(MODEL, "tokens", BUCKET_ID[1], 50.0)

    assert backend._bucket_registry[BUCKET_ID].max_capacity == 50.0
    assert _runtime_override(limiter) == 50.0


def test_set_max_capacity_reconciles_post_write_systemexit() -> None:
    outcome: dict[str, Any] = {}

    async def scenario() -> None:
        limiter = RateLimiter(_config(), backend=MemoryBackendBuilder())
        reservation = await limiter.acquire_capacity({"tokens": 1}, MODEL)
        backend = limiter._model_family_to_backend[MODEL_FAMILY]
        original_set_max_capacity = backend.set_max_capacity

        async def write_then_interrupt(
            metric: str,
            per_seconds: int,
            value: float,
        ) -> None:
            await original_set_max_capacity(metric, per_seconds, value)
            raise SystemExit("simulated set_max interrupt")

        backend.set_max_capacity = write_then_interrupt

        try:
            await limiter.set_max_capacity(MODEL, "tokens", BUCKET_ID[1], 50.0)
        except BaseException as exc:
            outcome["public_exception_type"] = type(exc)
        else:  # pragma: no cover - the backend write must interrupt set_max.
            outcome["public_exception_type"] = None

        outcome["backend_max_capacity"] = backend._bucket_registry[
            BUCKET_ID
        ].max_capacity
        outcome["runtime_override"] = _runtime_override(limiter)
        await limiter.refund_capacity({"tokens": 1}, reservation)
        await asyncio.wait_for(limiter.aclose(), timeout=1.0)

    with pytest.raises(SystemExit, match="simulated set_max interrupt"):
        asyncio.run(scenario())

    assert outcome["public_exception_type"] in {
        SystemExit,
        asyncio.CancelledError,
    }
    assert outcome["backend_max_capacity"] == pytest.approx(50.0)
    assert outcome["runtime_override"] == pytest.approx(50.0)
