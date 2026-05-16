from __future__ import annotations

import asyncio
import inspect
import time
import uuid
import warnings
from typing import TYPE_CHECKING, cast

from token_throttle._exceptions import BackendConformanceError
from token_throttle._interfaces._callbacks import (
    RateLimiterCallbacks,
    SyncRateLimiterCallbacks,
)
from token_throttle._interfaces._interfaces import (
    PerModelConfig,
    RateLimiterBackend,
    RateLimiterBackendBuilderInterface,
    SyncRateLimiterBackend,
    SyncRateLimiterBackendBuilderInterface,
    backend_uses_default_prepare_reconfigured_backend,
    backend_uses_default_refund_capacity_for_buckets,
    sync_backend_uses_default_prepare_reconfigured_backend,
    sync_backend_uses_default_refund_capacity_for_buckets,
)
from token_throttle._interfaces._models import Quota, UsageQuotas, frozen_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable

_IMMEDIATE_LIMIT_SECONDS = 1.0
_WAIT_TIMEOUT_SECONDS = 2.0
_TRY_ACQUIRE_TIMEOUT_SECONDS = 0.0
_SHORT_WINDOW_SECONDS = 1
_FAST_LIMIT = 10.0
_CALLBACK_LIMIT = 4.0
_RESERVATION_LIFETIME_SECONDS = 30.0
_REQUESTS_BUCKET_ID = ("requests", _SHORT_WINDOW_SECONDS)


def _family(label: str) -> str:
    return f"conformance/{label}/{uuid.uuid4().hex[:12]}"


def _config(
    label: str,
    *,
    limit: float = _FAST_LIMIT,
    extra_quotas: tuple[Quota, ...] = (),
) -> PerModelConfig:
    quotas = [
        Quota(
            metric="requests",
            limit=limit,
            per_seconds=_SHORT_WINDOW_SECONDS,
        )
    ]
    quotas.extend(extra_quotas)
    return PerModelConfig(
        model_family=_family(label),
        quotas=UsageQuotas(quotas),
    )


def _fail(message: str) -> None:
    raise BackendConformanceError(message)


def _check(condition: object, message: str) -> None:
    if not condition:
        _fail(message)


def _close_awaitable(value: object) -> None:
    if not inspect.isawaitable(value):
        return
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _check_bool_claim(value: object, method_name: str) -> bool:
    if inspect.isawaitable(value):
        _close_awaitable(value)
        _fail(f"{method_name}() must be synchronous and return bool")
    if type(value) is not bool:
        _fail(f"{method_name}() must return bool, got {type(value).__name__}")
    return cast("bool", value)


def _check_runtime_protocols(
    backend_builder: object,
    backend: object,
    *,
    sync: bool,
) -> None:
    if sync:
        _check(
            isinstance(backend_builder, SyncRateLimiterBackendBuilderInterface),
            "sync backend builder does not satisfy SyncRateLimiterBackendBuilderInterface",
        )
        _check(
            isinstance(backend, SyncRateLimiterBackend),
            "sync backend does not satisfy SyncRateLimiterBackend",
        )
        return

    _check(
        isinstance(backend_builder, RateLimiterBackendBuilderInterface),
        "async backend builder does not satisfy RateLimiterBackendBuilderInterface",
    )
    _check(
        isinstance(backend, RateLimiterBackend),
        "async backend does not satisfy RateLimiterBackend",
    )


def _check_async_claims(backend: RateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        backend.supports_acquire_marker_authority(),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        backend.supports_durable_refund_dedup(),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        backend.supports_metric_set_change(),
        "supports_metric_set_change",
    )

    if marker_authority and backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_acquire_marker_authority=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if durable_dedup and backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_durable_refund_dedup=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if metric_set_change and backend_uses_default_prepare_reconfigured_backend(backend):
        _fail(
            "supports_metric_set_change=True requires an override of "
            "prepare_reconfigured_backend()"
        )


def _check_sync_claims(backend: SyncRateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        backend.supports_acquire_marker_authority(),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        backend.supports_durable_refund_dedup(),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        backend.supports_metric_set_change(),
        "supports_metric_set_change",
    )

    if marker_authority and sync_backend_uses_default_refund_capacity_for_buckets(
        backend
    ):
        _fail(
            "supports_acquire_marker_authority=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if durable_dedup and sync_backend_uses_default_refund_capacity_for_buckets(backend):
        _fail(
            "supports_durable_refund_dedup=True requires an override of "
            "refund_capacity_for_buckets()"
        )
    if metric_set_change and sync_backend_uses_default_prepare_reconfigured_backend(
        backend
    ):
        _fail(
            "supports_metric_set_change=True requires an override of "
            "prepare_reconfigured_backend()"
        )


async def _expect_async_value_error(awaitable: Awaitable[object], message: str) -> None:
    try:
        await awaitable
    except ValueError:
        return
    _fail(message)


async def _expect_async_timeout(awaitable: Awaitable[object], message: str) -> None:
    try:
        await awaitable
    except TimeoutError:
        return
    _fail(message)


def _expect_value_error(fn, message: str) -> None:
    try:
        fn()
    except ValueError:
        return
    _fail(message)


def _expect_timeout(fn, message: str) -> None:
    try:
        fn()
    except TimeoutError:
        return
    _fail(message)


async def _check_async_basic_capacity(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = builder.build(_config("async-basic"))
    _check_async_claims(backend)

    start = time.monotonic()
    await backend.await_for_capacity(frozen_usage({"requests": 1}))
    _check(
        time.monotonic() - start < _IMMEDIATE_LIMIT_SECONDS,
        "await_for_capacity() did not return promptly when capacity was available",
    )

    await _expect_async_value_error(
        backend.await_for_capacity(frozen_usage({"requests": -1})),
        "await_for_capacity() must reject negative usage",
    )

    exhausted = builder.build(_config("async-exhaust"))
    await exhausted.await_for_capacity(frozen_usage({"requests": _FAST_LIMIT}))
    await _expect_async_timeout(
        exhausted.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "await_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    await exhausted.refund_capacity(
        frozen_usage({"requests": _FAST_LIMIT}),
        frozen_usage({"requests": _FAST_LIMIT / 2}),
    )
    start = time.monotonic()
    await exhausted.await_for_capacity(frozen_usage({"requests": _FAST_LIMIT / 2}))
    _check(
        time.monotonic() - start < _IMMEDIATE_LIMIT_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


async def _check_async_all_or_nothing(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = builder.build(
        _config(
            "async-all-or-nothing",
            limit=1.0,
            extra_quotas=(
                Quota(
                    metric="tokens",
                    limit=1.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ),
        )
    )
    await backend.await_for_capacity(frozen_usage({"tokens": 1, "requests": 0}))
    await _expect_async_timeout(
        backend.await_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "await_for_capacity() must not partially consume when one metric lacks capacity",
    )
    await backend.await_for_capacity(
        frozen_usage({"tokens": 0, "requests": 1}),
        timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
    )


async def _check_async_refund_and_overuse(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = builder.build(_config("async-invalid-refund"))
    await invalid_refund.await_for_capacity(frozen_usage({"requests": 1}))
    await _expect_async_value_error(
        invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = builder.build(_config("async-overuse"))
    await overuse.await_for_capacity(frozen_usage({"requests": 1}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await overuse.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 2}),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


async def _check_async_consume_and_capacity_updates(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    consume = builder.build(_config("async-consume", limit=5.0))
    await consume.await_for_capacity(frozen_usage({"requests": 5}))
    await consume.consume_capacity(frozen_usage({"requests": 5}))
    await _expect_async_timeout(
        consume.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = builder.build(_config("async-max-capacity", limit=5.0))
    await capacity.apply_configured_max_capacity("requests", _SHORT_WINDOW_SECONDS, 3.0)
    await _expect_async_value_error(
        capacity.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    await capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0)
    await capacity.await_for_capacity(
        frozen_usage({"requests": 4}),
        timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
    )


async def _check_async_callbacks(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    events: list[str] = []

    async def on_wait_start(**_kwargs) -> None:
        events.append("wait_start")

    async def after_wait_end_consumption(**_kwargs) -> None:
        events.append("wait_end")

    async def on_capacity_consumed(**_kwargs) -> None:
        events.append("consumed")

    async def on_capacity_refunded(**_kwargs) -> None:
        events.append("refunded")

    async def on_missing_consumption_data(**_kwargs) -> None:
        events.append("missing")

    backend = builder.build(
        _config("async-callbacks", limit=_CALLBACK_LIMIT),
        callbacks=RateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    await backend.await_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT}))
    await backend.await_for_capacity(
        frozen_usage({"requests": 1}),
        timeout=_WAIT_TIMEOUT_SECONDS,
    )
    await backend.refund_capacity(
        frozen_usage({"requests": 1}),
        frozen_usage({"requests": 0}),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


async def _check_async_marker_authority(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("async-marker")
    backend = builder.build(cfg)
    if not _check_bool_claim(
        backend.supports_acquire_marker_authority(),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    await backend.await_for_capacity(
        reserved_usage,
        reservation_id=reservation_id,
        reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
    )
    result = await backend.refund_capacity_for_buckets(
        reserved_usage,
        frozen_usage({"requests": 0}),
        bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
        reservation_id=reservation_id,
        reservation_model_family=cfg.get_model_family(),
        reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
        reservation_reserved_usage=reserved_usage,
    )
    _check(
        result is True,
        "refund_capacity_for_buckets() must return True after a marker-authorized refund",
    )


async def conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    """
    Run the public async backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.
    """
    probe_backend = backend_builder.build(_config("async-protocol-probe"))
    _check_runtime_protocols(backend_builder, probe_backend, sync=False)
    await _check_async_basic_capacity(backend_builder)
    await _check_async_all_or_nothing(backend_builder)
    await _check_async_refund_and_overuse(backend_builder)
    await _check_async_consume_and_capacity_updates(backend_builder)
    await _check_async_callbacks(backend_builder)
    await _check_async_marker_authority(backend_builder)


def _check_sync_basic_capacity(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = builder.build(_config("sync-basic"))
    _check_sync_claims(backend)

    start = time.monotonic()
    backend.wait_for_capacity(frozen_usage({"requests": 1}))
    _check(
        time.monotonic() - start < _IMMEDIATE_LIMIT_SECONDS,
        "wait_for_capacity() did not return promptly when capacity was available",
    )

    _expect_value_error(
        lambda: backend.wait_for_capacity(frozen_usage({"requests": -1})),
        "wait_for_capacity() must reject negative usage",
    )

    exhausted = builder.build(_config("sync-exhaust"))
    exhausted.wait_for_capacity(frozen_usage({"requests": _FAST_LIMIT}))
    _expect_timeout(
        lambda: exhausted.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    exhausted.refund_capacity(
        frozen_usage({"requests": _FAST_LIMIT}),
        frozen_usage({"requests": _FAST_LIMIT / 2}),
    )
    start = time.monotonic()
    exhausted.wait_for_capacity(frozen_usage({"requests": _FAST_LIMIT / 2}))
    _check(
        time.monotonic() - start < _IMMEDIATE_LIMIT_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


def _check_sync_all_or_nothing(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = builder.build(
        _config(
            "sync-all-or-nothing",
            limit=1.0,
            extra_quotas=(
                Quota(
                    metric="tokens",
                    limit=1.0,
                    per_seconds=_SHORT_WINDOW_SECONDS,
                ),
            ),
        )
    )
    backend.wait_for_capacity(frozen_usage({"tokens": 1, "requests": 0}))
    _expect_timeout(
        lambda: backend.wait_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity() must not partially consume when one metric lacks capacity",
    )
    backend.wait_for_capacity(
        frozen_usage({"tokens": 0, "requests": 1}),
        timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
    )


def _check_sync_refund_and_overuse(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = builder.build(_config("sync-invalid-refund"))
    invalid_refund.wait_for_capacity(frozen_usage({"requests": 1}))
    _expect_value_error(
        lambda: invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = builder.build(_config("sync-overuse"))
    overuse.wait_for_capacity(frozen_usage({"requests": 1}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overuse.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 2}),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


def _check_sync_consume_and_capacity_updates(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    consume = builder.build(_config("sync-consume", limit=5.0))
    consume.wait_for_capacity(frozen_usage({"requests": 5}))
    consume.consume_capacity(frozen_usage({"requests": 5}))
    _expect_timeout(
        lambda: consume.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = builder.build(_config("sync-max-capacity", limit=5.0))
    capacity.apply_configured_max_capacity("requests", _SHORT_WINDOW_SECONDS, 3.0)
    _expect_value_error(
        lambda: capacity.wait_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0)
    capacity.wait_for_capacity(
        frozen_usage({"requests": 4}),
        timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
    )


def _check_sync_callbacks(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    events: list[str] = []

    def on_wait_start(**_kwargs) -> None:
        events.append("wait_start")

    def after_wait_end_consumption(**_kwargs) -> None:
        events.append("wait_end")

    def on_capacity_consumed(**_kwargs) -> None:
        events.append("consumed")

    def on_capacity_refunded(**_kwargs) -> None:
        events.append("refunded")

    def on_missing_consumption_data(**_kwargs) -> None:
        events.append("missing")

    backend = builder.build(
        _config("sync-callbacks", limit=_CALLBACK_LIMIT),
        callbacks=SyncRateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    backend.wait_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT}))
    backend.wait_for_capacity(
        frozen_usage({"requests": 1}),
        timeout=_WAIT_TIMEOUT_SECONDS,
    )
    backend.refund_capacity(
        frozen_usage({"requests": 1}),
        frozen_usage({"requests": 0}),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


def _check_sync_marker_authority(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("sync-marker")
    backend = builder.build(cfg)
    if not _check_bool_claim(
        backend.supports_acquire_marker_authority(),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    backend.wait_for_capacity(
        reserved_usage,
        reservation_id=reservation_id,
        reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
    )
    result = backend.refund_capacity_for_buckets(
        reserved_usage,
        frozen_usage({"requests": 0}),
        bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
        reservation_id=reservation_id,
        reservation_model_family=cfg.get_model_family(),
        reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
        reservation_reserved_usage=reserved_usage,
    )
    _check(
        result is True,
        "refund_capacity_for_buckets() must return True after a marker-authorized refund",
    )


def sync_conformance_test_for(
    backend_builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    """
    Run the public sync backend conformance checks for one backend builder.

    The builder should point at isolated backend state: use a disposable Redis
    key prefix, database, or in-memory instance so these tests can consume and
    refund capacity freely.
    """
    probe_backend = backend_builder.build(_config("sync-protocol-probe"))
    _check_runtime_protocols(backend_builder, probe_backend, sync=True)
    _check_sync_basic_capacity(backend_builder)
    _check_sync_all_or_nothing(backend_builder)
    _check_sync_refund_and_overuse(backend_builder)
    _check_sync_consume_and_capacity_updates(backend_builder)
    _check_sync_callbacks(backend_builder)
    _check_sync_marker_authority(backend_builder)


def run_conformance_test_for(
    backend_builder: RateLimiterBackendBuilderInterface,
) -> None:
    """Run async backend conformance checks from synchronous test suites."""
    asyncio.run(conformance_test_for(backend_builder))
