from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
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
    from collections.abc import Awaitable, Callable

_BUILDER_DEADLINE_SECONDS = 5.0
_OPERATION_DEADLINE_SECONDS = 10.0
_PROMPT_DEADLINE_SECONDS = 1.0
_WAIT_BUDGET_SECONDS = 5.0
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


def _matches_allowed_exception(
    exc: BaseException,
    allowed_exceptions: tuple[type[BaseException], ...],
) -> bool:
    return bool(allowed_exceptions) and isinstance(exc, allowed_exceptions)


def _run_sync_step(
    label: str,
    callable_: Callable[[], object],
    *,
    deadline: float,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    """
    Run a synchronous callable under a wall-clock deadline; normalize exceptions.

    KNOWN LIMITATION: Python cannot safely kill an in-process thread that
    ignores its task. A timeout reports the hang and shuts down the executor
    without waiting, but the worker thread may continue running.
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="token-throttle-conformance",
    )
    future = executor.submit(callable_)
    timed_out = False
    try:
        return future.result(timeout=deadline)
    except TimeoutError as exc:
        if future.done():
            if _matches_allowed_exception(exc, allowed_exceptions):
                raise
            raise BackendConformanceError(
                f"{label} raised {type(exc).__name__}: {exc}"
            ) from exc
        timed_out = True
        message = (
            f"{label} did not return within {deadline}s; thread may still be "
            "running (cannot kill in-process)"
        )
        warnings.warn(message, RuntimeWarning)
        raise BackendConformanceError(message) from exc
    except BackendConformanceError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if _matches_allowed_exception(exc, allowed_exceptions):
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        executor.shutdown(wait=not timed_out, cancel_futures=True)


def _task_is_being_cancelled() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _consume_abandoned_task_result(task: asyncio.Future[object]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


async def _run_async_step(
    label: str,
    awaitable_or_coro_fn: Awaitable[object] | Callable[[], object] | object,
    *,
    deadline: float,
    expect_awaitable: bool = False,
    allowed_exceptions: tuple[type[BaseException], ...] = (),
) -> object:
    """Run an awaitable under a wall-clock deadline; normalize exceptions."""
    try:
        value: object
        if inspect.isawaitable(awaitable_or_coro_fn):
            value = awaitable_or_coro_fn
        elif callable(awaitable_or_coro_fn):
            value = _run_sync_step(
                label,
                awaitable_or_coro_fn,
                deadline=deadline,
                allowed_exceptions=allowed_exceptions,
            )
        else:
            value = awaitable_or_coro_fn

        if not inspect.isawaitable(value):
            if expect_awaitable:
                _fail(f"{label} returned non-awaitable {type(value).__name__}")
            return value

        task = asyncio.ensure_future(cast("Awaitable[object]", value))
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
        except TimeoutError as exc:
            if task.done() and not task.cancelled():
                if _matches_allowed_exception(exc, allowed_exceptions):
                    raise
                raise BackendConformanceError(
                    f"{label} raised {type(exc).__name__}: {exc}"
                ) from exc
            task.add_done_callback(_consume_abandoned_task_result)
            task.cancel()
            raise BackendConformanceError(
                f"{label} did not return within {deadline}s"
            ) from exc
    except BackendConformanceError:
        raise
    except asyncio.CancelledError as exc:
        if _task_is_being_cancelled():
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if _matches_allowed_exception(exc, allowed_exceptions):
            raise
        raise BackendConformanceError(
            f"{label} raised {type(exc).__name__}: {exc}"
        ) from exc


def _close_awaitable(value: object) -> None:
    if not inspect.isawaitable(value):
        return
    close = getattr(value, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
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


def _build_async_backend(
    builder: RateLimiterBackendBuilderInterface,
    cfg: PerModelConfig,
    *,
    label: str,
    callbacks: RateLimiterCallbacks | None = None,
) -> RateLimiterBackend:
    backend = _run_sync_step(
        label,
        lambda: builder.build(cfg, callbacks=callbacks),
        deadline=_BUILDER_DEADLINE_SECONDS,
    )
    _check_runtime_protocols(builder, backend, sync=False)
    return cast("RateLimiterBackend", backend)


def _build_sync_backend(
    builder: SyncRateLimiterBackendBuilderInterface,
    cfg: PerModelConfig,
    *,
    label: str,
    callbacks: SyncRateLimiterCallbacks | None = None,
) -> SyncRateLimiterBackend:
    backend = _run_sync_step(
        label,
        lambda: builder.build(cfg, callbacks=callbacks),
        deadline=_BUILDER_DEADLINE_SECONDS,
    )
    _check_runtime_protocols(builder, backend, sync=True)
    return cast("SyncRateLimiterBackend", backend)


async def _async_backend_step(
    label: str,
    awaitable_fn: Callable[[], object],
    *,
    deadline: float | None = None,
) -> object:
    return await _run_async_step(
        label,
        awaitable_fn,
        deadline=_OPERATION_DEADLINE_SECONDS if deadline is None else deadline,
        expect_awaitable=True,
    )


def _sync_backend_step(
    label: str,
    callable_: Callable[[], object],
    *,
    deadline: float | None = None,
) -> object:
    return _run_sync_step(
        label,
        callable_,
        deadline=_OPERATION_DEADLINE_SECONDS if deadline is None else deadline,
    )


def _check_async_claims(backend: RateLimiterBackend) -> None:
    marker_authority = _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
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
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    )
    durable_dedup = _check_bool_claim(
        _run_sync_step(
            "supports_durable_refund_dedup()",
            backend.supports_durable_refund_dedup,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_durable_refund_dedup",
    )
    metric_set_change = _check_bool_claim(
        _run_sync_step(
            "supports_metric_set_change()",
            backend.supports_metric_set_change,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
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


async def _expect_async_value_error(
    label: str,
    awaitable_fn: Callable[[], object],
    message: str,
) -> None:
    try:
        await _run_async_step(
            label,
            awaitable_fn,
            deadline=_OPERATION_DEADLINE_SECONDS,
            expect_awaitable=True,
            allowed_exceptions=(ValueError,),
        )
    except ValueError:
        return
    _fail(message)


async def _expect_async_timeout(
    label: str,
    awaitable_fn: Callable[[], object],
    message: str,
    *,
    promptness_deadline: float = _PROMPT_DEADLINE_SECONDS,
) -> None:
    start = time.monotonic()
    try:
        await _run_async_step(
            label,
            awaitable_fn,
            deadline=_WAIT_BUDGET_SECONDS,
            expect_awaitable=True,
            allowed_exceptions=(TimeoutError,),
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        if elapsed < promptness_deadline:
            return
        _fail(
            f"{label} raised TimeoutError after {elapsed:.2f}s; "
            "expected prompt try-acquire"
        )
    _fail(message)


def _expect_value_error(
    label: str,
    fn: Callable[[], object],
    message: str,
) -> None:
    try:
        _run_sync_step(
            label,
            fn,
            deadline=_OPERATION_DEADLINE_SECONDS,
            allowed_exceptions=(ValueError,),
        )
    except ValueError:
        return
    _fail(message)


def _expect_timeout(
    label: str,
    fn: Callable[[], object],
    message: str,
    *,
    promptness_deadline: float = _PROMPT_DEADLINE_SECONDS,
) -> None:
    start = time.monotonic()
    try:
        _run_sync_step(
            label,
            fn,
            deadline=_WAIT_BUDGET_SECONDS,
            allowed_exceptions=(TimeoutError,),
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        if elapsed < promptness_deadline:
            return
        _fail(
            f"{label} raised TimeoutError after {elapsed:.2f}s; "
            "expected prompt try-acquire"
        )
    _fail(message)


async def _check_async_basic_capacity(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_async_backend(
        builder,
        _config("async-basic"),
        label="build(async-basic)",
    )
    _check_async_claims(backend)

    start = time.monotonic()
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": 1})),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "await_for_capacity() did not return promptly when capacity was available",
    )

    await _expect_async_value_error(
        "await_for_capacity(requests=-1)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": -1})),
        "await_for_capacity() must reject negative usage",
    )

    exhausted = _build_async_backend(
        builder,
        _config("async-exhaust"),
        label="build(async-exhaust)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=10)",
        lambda: exhausted.await_for_capacity(frozen_usage({"requests": _FAST_LIMIT})),
    )
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: exhausted.await_for_capacity(
            frozen_usage({"requests": 1}), timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS
        ),
        "await_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    await _async_backend_step(
        "refund_capacity(requests=10, actual=5)",
        lambda: exhausted.refund_capacity(
            frozen_usage({"requests": _FAST_LIMIT}),
            frozen_usage({"requests": _FAST_LIMIT / 2}),
        ),
    )
    start = time.monotonic()
    await _async_backend_step(
        "await_for_capacity(requests=5)",
        lambda: exhausted.await_for_capacity(
            frozen_usage({"requests": _FAST_LIMIT / 2})
        ),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


async def _check_async_all_or_nothing(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_async_backend(
        builder,
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
        ),
        label="build(async-all-or-nothing)",
    )
    await _async_backend_step(
        "await_for_capacity(tokens=1, requests=0)",
        lambda: backend.await_for_capacity(frozen_usage({"tokens": 1, "requests": 0})),
    )
    await _expect_async_timeout(
        "await_for_capacity(tokens=1, requests=1, timeout=0)",
        lambda: backend.await_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "await_for_capacity() must not partially consume when one metric lacks capacity",
    )
    await _async_backend_step(
        "await_for_capacity(tokens=0, requests=1, timeout=0)",
        lambda: backend.await_for_capacity(
            frozen_usage({"tokens": 0, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


async def _check_async_refund_and_overuse(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = _build_async_backend(
        builder,
        _config("async-invalid-refund"),
        label="build(async-invalid-refund)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: invalid_refund.await_for_capacity(frozen_usage({"requests": 1})),
    )
    await _expect_async_value_error(
        "refund_capacity(requests=1, actual=-1)",
        lambda: invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = _build_async_backend(
        builder,
        _config("async-overuse"),
        label="build(async-overuse)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=1)",
        lambda: overuse.await_for_capacity(frozen_usage({"requests": 1})),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await _async_backend_step(
            "refund_capacity(requests=1, actual=2)",
            lambda: overuse.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 2}),
            ),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


async def _check_async_consume_and_capacity_updates(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    consume = _build_async_backend(
        builder,
        _config("async-consume", limit=5.0),
        label="build(async-consume)",
    )
    await _async_backend_step(
        "await_for_capacity(requests=5)",
        lambda: consume.await_for_capacity(frozen_usage({"requests": 5})),
    )
    await _async_backend_step(
        "consume_capacity(requests=5)",
        lambda: consume.consume_capacity(frozen_usage({"requests": 5})),
    )
    await _expect_async_timeout(
        "await_for_capacity(requests=1, timeout=0)",
        lambda: consume.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = _build_async_backend(
        builder,
        _config("async-max-capacity", limit=5.0),
        label="build(async-max-capacity)",
    )
    await _async_backend_step(
        "apply_configured_max_capacity(requests, 1, 3)",
        lambda: capacity.apply_configured_max_capacity(
            "requests", _SHORT_WINDOW_SECONDS, 3.0
        ),
    )
    await _expect_async_value_error(
        "await_for_capacity(requests=4, timeout=0)",
        lambda: capacity.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    await _async_backend_step(
        "set_max_capacity(requests, 1, 4)",
        lambda: capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0),
    )
    await _async_backend_step(
        "await_for_capacity(requests=4, timeout=0)",
        lambda: capacity.await_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
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

    backend = _build_async_backend(
        builder,
        _config("async-callbacks", limit=_CALLBACK_LIMIT),
        label="build(async-callbacks)",
        callbacks=RateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    await _async_backend_step(
        "await_for_capacity(requests=4)",
        lambda: backend.await_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT})),
    )
    await _async_backend_step(
        "await_for_capacity(requests=1, timeout=2)",
        lambda: backend.await_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_WAIT_TIMEOUT_SECONDS,
        ),
        deadline=_WAIT_BUDGET_SECONDS,
    )
    await _async_backend_step(
        "refund_capacity(requests=1, actual=0)",
        lambda: backend.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 0}),
        ),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


async def _check_async_marker_authority(
    builder: RateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("async-marker")
    backend = _build_async_backend(
        builder,
        cfg,
        label="build(async-marker)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    await _async_backend_step(
        "await_for_capacity(requests=1, reservation_id=...)",
        lambda: backend.await_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    result = await _async_backend_step(
        "refund_capacity_for_buckets(requests=1, actual=0)",
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            frozen_usage({"requests": 0}),
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_reserved_usage=reserved_usage,
        ),
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

    Backend operations are bounded by helper-owned deadlines. KNOWN LIMITATION:
    if a synchronous backend call hangs in its worker thread, Python cannot
    safely kill that thread; the helper reports the hang and continues without
    waiting for that thread to finish.
    """
    _build_async_backend(
        backend_builder,
        _config("async-protocol-probe"),
        label="build(async-protocol-probe)",
    )
    await _check_async_basic_capacity(backend_builder)
    await _check_async_all_or_nothing(backend_builder)
    await _check_async_refund_and_overuse(backend_builder)
    await _check_async_consume_and_capacity_updates(backend_builder)
    await _check_async_callbacks(backend_builder)
    await _check_async_marker_authority(backend_builder)


def _check_sync_basic_capacity(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_sync_backend(
        builder,
        _config("sync-basic"),
        label="build(sync-basic)",
    )
    _check_sync_claims(backend)

    start = time.monotonic()
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "wait_for_capacity() did not return promptly when capacity was available",
    )

    _expect_value_error(
        "wait_for_capacity(requests=-1)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": -1})),
        "wait_for_capacity() must reject negative usage",
    )

    exhausted = _build_sync_backend(
        builder,
        _config("sync-exhaust"),
        label="build(sync-exhaust)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=10)",
        lambda: exhausted.wait_for_capacity(frozen_usage({"requests": _FAST_LIMIT})),
    )
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: exhausted.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity(timeout=0) must raise TimeoutError when capacity is unavailable",
    )
    _sync_backend_step(
        "refund_capacity(requests=10, actual=5)",
        lambda: exhausted.refund_capacity(
            frozen_usage({"requests": _FAST_LIMIT}),
            frozen_usage({"requests": _FAST_LIMIT / 2}),
        ),
    )
    start = time.monotonic()
    _sync_backend_step(
        "wait_for_capacity(requests=5)",
        lambda: exhausted.wait_for_capacity(
            frozen_usage({"requests": _FAST_LIMIT / 2})
        ),
    )
    _check(
        time.monotonic() - start < _PROMPT_DEADLINE_SECONDS,
        "refund_capacity() did not restore unused capacity",
    )


def _check_sync_all_or_nothing(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    backend = _build_sync_backend(
        builder,
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
        ),
        label="build(sync-all-or-nothing)",
    )
    _sync_backend_step(
        "wait_for_capacity(tokens=1, requests=0)",
        lambda: backend.wait_for_capacity(frozen_usage({"tokens": 1, "requests": 0})),
    )
    _expect_timeout(
        "wait_for_capacity(tokens=1, requests=1, timeout=0)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"tokens": 1, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "wait_for_capacity() must not partially consume when one metric lacks capacity",
    )
    _sync_backend_step(
        "wait_for_capacity(tokens=0, requests=1, timeout=0)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"tokens": 0, "requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
    )


def _check_sync_refund_and_overuse(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    invalid_refund = _build_sync_backend(
        builder,
        _config("sync-invalid-refund"),
        label="build(sync-invalid-refund)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: invalid_refund.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    _expect_value_error(
        "refund_capacity(requests=1, actual=-1)",
        lambda: invalid_refund.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": -1}),
        ),
        "refund_capacity() must reject negative actual usage",
    )

    overuse = _build_sync_backend(
        builder,
        _config("sync-overuse"),
        label="build(sync-overuse)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1)",
        lambda: overuse.wait_for_capacity(frozen_usage({"requests": 1})),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _sync_backend_step(
            "refund_capacity(requests=1, actual=2)",
            lambda: overuse.refund_capacity(
                frozen_usage({"requests": 1}),
                frozen_usage({"requests": 2}),
            ),
        )
    _check(
        any(issubclass(item.category, RuntimeWarning) for item in caught),
        "refund_capacity() must warn with RuntimeWarning when actual usage exceeds reserved usage",
    )


def _check_sync_consume_and_capacity_updates(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    consume = _build_sync_backend(
        builder,
        _config("sync-consume", limit=5.0),
        label="build(sync-consume)",
    )
    _sync_backend_step(
        "wait_for_capacity(requests=5)",
        lambda: consume.wait_for_capacity(frozen_usage({"requests": 5})),
    )
    _sync_backend_step(
        "consume_capacity(requests=5)",
        lambda: consume.consume_capacity(frozen_usage({"requests": 5})),
    )
    _expect_timeout(
        "wait_for_capacity(requests=1, timeout=0)",
        lambda: consume.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "consume_capacity() must allow debt that later blocking acquires observe",
    )

    capacity = _build_sync_backend(
        builder,
        _config("sync-max-capacity", limit=5.0),
        label="build(sync-max-capacity)",
    )
    _sync_backend_step(
        "apply_configured_max_capacity(requests, 1, 3)",
        lambda: capacity.apply_configured_max_capacity(
            "requests", _SHORT_WINDOW_SECONDS, 3.0
        ),
    )
    _expect_value_error(
        "wait_for_capacity(requests=4, timeout=0)",
        lambda: capacity.wait_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
        "apply_configured_max_capacity() must update the live bucket max_capacity",
    )
    _sync_backend_step(
        "set_max_capacity(requests, 1, 4)",
        lambda: capacity.set_max_capacity("requests", _SHORT_WINDOW_SECONDS, 4.0),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=4, timeout=0)",
        lambda: capacity.wait_for_capacity(
            frozen_usage({"requests": 4}),
            timeout=_TRY_ACQUIRE_TIMEOUT_SECONDS,
        ),
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

    backend = _build_sync_backend(
        builder,
        _config("sync-callbacks", limit=_CALLBACK_LIMIT),
        label="build(sync-callbacks)",
        callbacks=SyncRateLimiterCallbacks(
            on_wait_start=on_wait_start,
            after_wait_end_consumption=after_wait_end_consumption,
            on_capacity_consumed=on_capacity_consumed,
            on_capacity_refunded=on_capacity_refunded,
            on_missing_consumption_data=on_missing_consumption_data,
        ),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=4)",
        lambda: backend.wait_for_capacity(frozen_usage({"requests": _CALLBACK_LIMIT})),
    )
    _sync_backend_step(
        "wait_for_capacity(requests=1, timeout=2)",
        lambda: backend.wait_for_capacity(
            frozen_usage({"requests": 1}),
            timeout=_WAIT_TIMEOUT_SECONDS,
        ),
        deadline=_WAIT_BUDGET_SECONDS,
    )
    _sync_backend_step(
        "refund_capacity(requests=1, actual=0)",
        lambda: backend.refund_capacity(
            frozen_usage({"requests": 1}),
            frozen_usage({"requests": 0}),
        ),
    )

    for event in ("missing", "consumed", "wait_start", "wait_end", "refunded"):
        _check(event in events, f"callback event {event!r} was not emitted")


def _check_sync_marker_authority(
    builder: SyncRateLimiterBackendBuilderInterface,
) -> None:
    cfg = _config("sync-marker")
    backend = _build_sync_backend(
        builder,
        cfg,
        label="build(sync-marker)",
    )
    if not _check_bool_claim(
        _run_sync_step(
            "supports_acquire_marker_authority()",
            backend.supports_acquire_marker_authority,
            deadline=_BUILDER_DEADLINE_SECONDS,
        ),
        "supports_acquire_marker_authority",
    ):
        return

    reservation_id = f"conformance-{uuid.uuid4().hex}"
    reserved_usage = frozen_usage({"requests": 1})
    _sync_backend_step(
        "wait_for_capacity(requests=1, reservation_id=...)",
        lambda: backend.wait_for_capacity(
            reserved_usage,
            reservation_id=reservation_id,
            reservation_lifetime_seconds=_RESERVATION_LIFETIME_SECONDS,
        ),
    )
    result = _sync_backend_step(
        "refund_capacity_for_buckets(requests=1, actual=0)",
        lambda: backend.refund_capacity_for_buckets(
            reserved_usage,
            frozen_usage({"requests": 0}),
            bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_id=reservation_id,
            reservation_model_family=cfg.get_model_family(),
            reservation_bucket_ids=frozenset({_REQUESTS_BUCKET_ID}),
            reservation_reserved_usage=reserved_usage,
        ),
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

    Backend operations are bounded by helper-owned deadlines. KNOWN LIMITATION:
    if a synchronous backend call hangs in its worker thread, Python cannot
    safely kill that thread; the helper reports the hang and continues without
    waiting for that thread to finish.
    """
    _build_sync_backend(
        backend_builder,
        _config("sync-protocol-probe"),
        label="build(sync-protocol-probe)",
    )
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
