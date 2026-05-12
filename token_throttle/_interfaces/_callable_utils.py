import asyncio
import inspect


def is_async_callable(value: object) -> bool:
    if inspect.iscoroutinefunction(value) or inspect.isasyncgenfunction(value):
        return True
    if not callable(value):
        return False
    return inspect.iscoroutinefunction(value.__call__) or inspect.isasyncgenfunction(
        value.__call__
    )


def close_awaitable_if_possible(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def suppress_current_task_cancellation() -> None:
    """
    Clear pending cancellation requests on the current task, if any.

    Rationale (speedometer semantics)
    ---------------------------------
    ``consume_capacity`` / ``record_usage`` record work that has *already
    happened* (tokens the provider actually billed, requests already sent).
    Once the Redis write has landed or the in-memory state has mutated, the
    recorded consumption is the correct reading of the speedometer. If a
    ``CancelledError`` then arrives while callbacks fire or while we
    observe the shielded write, refunding would roll back a measurement
    of real usage and understate throughput — that's wrong for this
    accounting model.

    We therefore uncancel the current task at sites where consumption has
    already been durably recorded. The trade-off is that a caller's
    ``TaskGroup`` cancellation can be silently absorbed here; that is the
    intentional cost of preserving the speedometer invariant. Acquire
    paths never call this helper — there, CancelledError propagates and
    the reserved capacity is refunded.
    """
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()
