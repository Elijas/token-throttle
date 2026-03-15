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
