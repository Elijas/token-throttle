import asyncio
import inspect


def is_async_callable(value: object) -> bool:
    if inspect.iscoroutinefunction(value):
        return True
    if not callable(value):
        return False
    return inspect.iscoroutinefunction(value.__call__)


def close_awaitable_if_possible(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def suppress_current_task_cancellation() -> None:
    """Clear pending cancellation requests on the current task, if any."""
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()
