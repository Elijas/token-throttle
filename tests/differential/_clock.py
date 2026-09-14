"""
One controlled wall clock for all six built-in backends.

* Memory backends read ``time.time()`` through their module-level ``time``
  import; we swap that module attribute for a proxy whose ``time()`` returns
  the fake clock and whose every other attribute is the real ``time`` module
  (``monotonic``/``sleep`` stay real so deadlines and polling keep working).
* SQLite engines accept ``clock=`` on every state-changing method but the
  backends never pass it; we bind ``clock=fake.time`` onto the engine
  instance so both the async and sync backends pick it up.
* Redis backends read the server clock via ``async_server_time`` /
  ``sync_server_time``; we replace those names in the modules that imported
  them. This also bypasses the forward-jump rail, which is the point: the
  harness moves the clock deliberately.

Redis key TTLs (``EX``/``PX``) and the per-process override cache still use
real time; fake-clock scenarios therefore use very long TTLs, and TTL-expiry
scenarios use the real clock with short TTLs.
"""

from __future__ import annotations

import contextlib
import functools
import time as _real_time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


class FakeClock:
    """Manually advanced wall clock; starts at the real ``time.time()``."""

    def __init__(self, start: float | None = None) -> None:
        self.now: float = _real_time.time() if start is None else float(start)

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class RealClock(FakeClock):
    """Drop-in for ``FakeClock`` that always reads the real wall clock.

    Use when a scenario must run on real time (TTL expiry) but goes through
    ``build_one``, which binds the clock into the SQLite engine unconditionally.
    """

    def __init__(self) -> None:
        super().__init__(_real_time.time())

    def time(self) -> float:
        return _real_time.time()

    def advance(self, seconds: float) -> None:
        raise RuntimeError("RealClock cannot be advanced")


class _TimeProxy:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.time()

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


_SQLITE_CLOCKED_METHODS = (
    "initialize_buckets",
    "try_consume",
    "consume",
    "refund",
    "cleanup_consumption",
    "set_max_capacity",
    "apply_configured_max_capacity",
    "clear_max_capacity_overrides",
    "inspect_snapshot",
)


def bind_sqlite_engine_clock(engine: object, clock: FakeClock) -> None:
    """Make every clock-accepting engine method default to the fake clock."""
    for name in _SQLITE_CLOCKED_METHODS:
        unbound = getattr(type(engine), name)
        setattr(engine, name, functools.partial(unbound, engine, clock=clock.time))


@contextlib.contextmanager
def patched_clock(clock: FakeClock) -> Iterator[None]:
    """Route the memory and Redis backends' wall-clock reads to ``clock``."""
    import token_throttle._limiter_backends._memory._backend as memory_async  # noqa: PLC0415
    import token_throttle._limiter_backends._memory._sync_backend as memory_sync  # noqa: PLC0415
    import token_throttle._limiter_backends._redis._backend as redis_async  # noqa: PLC0415
    import token_throttle._limiter_backends._redis._bucket as redis_bucket  # noqa: PLC0415
    import token_throttle._limiter_backends._redis._sync_backend as redis_sync  # noqa: PLC0415
    import token_throttle._limiter_backends._redis._sync_bucket as redis_sync_bucket  # noqa: PLC0415

    async def fake_async_server_time(_client: object) -> float:
        return clock.time()

    def fake_sync_server_time(_client: object) -> float:
        return clock.time()

    proxy = _TimeProxy(clock)
    patches: list[tuple[Any, str, Any]] = [
        (memory_async, "time", proxy),
        (memory_sync, "time", proxy),
        (redis_async, "async_server_time", fake_async_server_time),
        (redis_bucket, "async_server_time", fake_async_server_time),
        (redis_sync, "sync_server_time", fake_sync_server_time),
        (redis_sync_bucket, "sync_server_time", fake_sync_server_time),
    ]
    saved = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, value in patches:
        setattr(module, name, value)
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)
