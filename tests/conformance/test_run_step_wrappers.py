from __future__ import annotations

import asyncio
import contextlib
import time
from typing import NoReturn

import pytest

from token_throttle import BackendConformanceError, conformance


def _raise(exc: BaseException) -> NoReturn:
    raise exc


def _flatten_group(exc: BaseException) -> list[BaseException]:
    if not isinstance(exc, BaseExceptionGroup):
        return [exc]
    leaves: list[BaseException] = []
    for nested in exc.exceptions:
        leaves.extend(_flatten_group(nested))
    return leaves


class _CancelledCloseAwaitable:
    def __await__(self):
        if False:
            yield None

    def close(self) -> None:
        raise asyncio.CancelledError("close cancelled")


async def test_external_cancellation_cancels_and_consumes_backend_task() -> None:
    loop = asyncio.get_running_loop()
    captured_contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def record_exception(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, object],
    ) -> None:
        captured_contexts.append(context)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def backend_step() -> None:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise RuntimeError("backend cancellation cleanup failed") from None

    loop.set_exception_handler(record_exception)
    try:
        wrapper_task = asyncio.create_task(
            conformance._run_async_step(
                "outer-cancel",
                backend_step(),
                deadline=1.0,
                expect_awaitable=True,
            )
        )
        await started.wait()

        wrapper_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wrapper_task

        await asyncio.wait_for(cancelled.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert captured_contexts == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_async_callable_materialization_does_not_block_event_loop() -> None:
    def blocking_materialization() -> object:
        time.sleep(0.25)
        return object()

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await conformance._run_async_step(
                "blocking-materialization",
                blocking_materialization,
                deadline=1.0,
            )
    elapsed = time.monotonic() - start

    assert elapsed < 0.2
    await asyncio.sleep(0.25)


async def test_async_step_uses_single_deadline_for_materialization_and_await() -> None:
    def materialize_slow_awaitable() -> object:
        time.sleep(0.07)

        async def slow_await_phase() -> None:
            await asyncio.sleep(0.08)

        return slow_await_phase()

    start = time.monotonic()
    with pytest.raises(
        BackendConformanceError,
        match=r"two-phase did not return within 0\.12s",
    ):
        await conformance._run_async_step(
            "two-phase",
            materialize_slow_awaitable,
            deadline=0.12,
            expect_awaitable=True,
        )
    elapsed = time.monotonic() - start

    assert elapsed < 0.17


def test_close_awaitable_suppresses_cancelled_error_from_close() -> None:
    conformance._close_awaitable(_CancelledCloseAwaitable())


async def test_close_awaitable_cancels_and_consumes_pending_future() -> None:
    loop = asyncio.get_running_loop()
    captured_contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def record_exception(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, object],
    ) -> None:
        captured_contexts.append(context)

    cancelled = asyncio.Event()

    async def backend_task() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise RuntimeError("cancel cleanup failed") from None

    loop.set_exception_handler(record_exception)
    try:
        task = asyncio.create_task(backend_task())
        future = loop.create_future()
        await asyncio.sleep(0)

        conformance._close_awaitable(task)
        conformance._close_awaitable(future)

        assert future.cancelled()
        await asyncio.wait_for(cancelled.wait(), timeout=0.5)
        await asyncio.sleep(0)
        assert task.done()
        assert captured_contexts == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_exception_group_control_leaves_propagate_before_normalization() -> None:
    async def grouped_failure() -> None:
        await asyncio.sleep(0)
        raise BaseExceptionGroup(
            "mixed",
            [asyncio.CancelledError("backend cancelled"), ValueError("bad")],
        )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await conformance._run_async_step(
            "grouped-step",
            grouped_failure(),
            deadline=0.2,
            expect_awaitable=True,
        )

    leaves = _flatten_group(exc_info.value)
    assert any(isinstance(leaf, asyncio.CancelledError) for leaf in leaves)
    assert any(isinstance(leaf, BackendConformanceError) for leaf in leaves)


async def test_allowed_exception_groups_are_normalized_as_grouped_failures() -> None:
    async def grouped_value_error() -> None:
        await asyncio.sleep(0)
        raise ExceptionGroup("grouped value error", [ValueError("expected")])

    with pytest.raises(
        BackendConformanceError,
        match="allowed-group raised ExceptionGroup",
    ):
        await conformance._run_async_step(
            "allowed-group",
            grouped_value_error(),
            deadline=0.2,
            expect_awaitable=True,
            allowed_exceptions=(ValueError,),
        )


async def test_backend_cancelled_error_is_not_confused_with_stale_cancellation() -> (
    None
):
    current_task = asyncio.current_task()
    assert current_task is not None
    current_task.cancel()
    try:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert current_task.cancelling() > 0

        async def backend_cancel() -> None:
            await asyncio.sleep(0)
            raise asyncio.CancelledError("backend-owned")

        with pytest.raises(
            BackendConformanceError,
            match="backend-cancel raised CancelledError",
        ):
            await conformance._run_async_step(
                "backend-cancel",
                backend_cancel(),
                deadline=0.2,
                expect_awaitable=True,
            )
    finally:
        while current_task.cancelling():
            current_task.uncancel()


async def test_backend_conformance_error_is_labeled_and_chained() -> None:
    sync_backend_error = BackendConformanceError("sync controlled")
    with pytest.raises(
        BackendConformanceError,
        match="sync-phase raised BackendConformanceError: sync controlled",
    ) as sync_exc:
        conformance._run_sync_step(
            "sync-phase",
            lambda: _raise(sync_backend_error),
            deadline=0.2,
        )

    assert sync_exc.value.__cause__ is sync_backend_error

    async_backend_error = BackendConformanceError("async controlled")

    async def async_failure() -> None:
        await asyncio.sleep(0)
        raise async_backend_error

    with pytest.raises(
        BackendConformanceError,
        match="async-phase raised BackendConformanceError: async controlled",
    ) as async_exc:
        await conformance._run_async_step(
            "async-phase",
            async_failure(),
            deadline=0.2,
            expect_awaitable=True,
        )

    assert async_exc.value.__cause__ is async_backend_error


@pytest.mark.parametrize("exc_type", [MemoryError, GeneratorExit])
def test_sync_step_reraises_catastrophic_exceptions(
    exc_type: type[BaseException],
) -> None:
    with pytest.raises(exc_type):
        conformance._run_sync_step(
            "catastrophic-sync",
            lambda: _raise(exc_type("catastrophic")),
            deadline=0.2,
        )


@pytest.mark.parametrize("exc_type", [MemoryError, GeneratorExit])
async def test_async_step_reraises_catastrophic_exceptions(
    exc_type: type[BaseException],
) -> None:
    async def catastrophic_failure() -> None:
        await asyncio.sleep(0)
        raise exc_type("catastrophic")

    with pytest.raises(exc_type):
        await conformance._run_async_step(
            "catastrophic-async",
            catastrophic_failure(),
            deadline=0.2,
            expect_awaitable=True,
        )
