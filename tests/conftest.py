from __future__ import annotations

import os
import threading
import time

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--redis-url",
        default="redis://localhost:6379",
        help="Redis URL for integration tests (default: redis://localhost:6379)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "redis: requires a running Redis instance")
    config.addinivalue_line("markers", "slow: long-running tests")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-mark tests that need Redis with the 'redis' marker.

    Also skip tests that directly use redis_client when parameterized
    with the memory backend (those tests are Redis-specific by nature).
    """
    for item in items:
        if "integration" not in str(item.fspath):
            continue

        # Skip tests that directly depend on redis_client when running
        # under the memory backend — they manipulate Redis keys that the
        # memory backend doesn't read.
        if "[memory" in item.nodeid and "redis_client" in item.fixturenames:
            item.add_marker(
                pytest.mark.skip(
                    reason="Test uses redis_client directly; "
                    "not applicable to memory backend",
                )
            )
            continue

        # Redis-specific test files always need Redis
        if "redis_specific" in str(item.fspath):
            item.add_marker(pytest.mark.redis)
            continue
        # Parameterized tests: only mark the [redis] variant
        if "[redis" in item.nodeid:
            item.add_marker(pytest.mark.redis)
        elif "[memory" not in item.nodeid:
            # Non-parameterized integration test — needs redis
            item.add_marker(pytest.mark.redis)


# ---------------------------------------------------------------------------
# Thread-leak detection
# ---------------------------------------------------------------------------
#
# The suite spawns many ``ThreadPoolExecutor`` instances (sync limiters, sync
# conformance steps, sync concurrency tests). A test that fails to join or shut
# down its executor leaks worker threads that outlive it. Accumulated leaked
# threads compete for the GIL and can stretch lock-hold windows, which is the
# leading hypothesis for the cross-test interference that made the Redis stress
# tests flake only in full-suite runs.
#
# This detector snapshots live threads at session start and re-checks at session
# finish (after a short grace period so naturally-finishing executor threads are
# not false positives). A thread is reported as a leak when it is:
#   (a) not present in the start snapshot,
#   (b) still alive after the grace period, and
#   (c) non-daemon OR a ThreadPoolExecutor worker (executor workers are daemonic
#       in current CPython, so a daemon-only filter would miss them; we key on
#       the executor thread-name prefixes instead).
#
# The chosen severity policy is documented in DEVELOPMENT.md. Setting
# ``TOKEN_THROTTLE_THREAD_LEAK_MODE=report`` forces non-failing REPORT mode for
# local debugging; the default (STRICT) fails the session on any leak.

_THREAD_LEAK_MODE_ENV = "TOKEN_THROTTLE_THREAD_LEAK_MODE"
_THREAD_LEAK_GRACE_SECONDS = 5.0
_THREAD_LEAK_POLL_SECONDS = 0.1

# Thread-name prefixes used by ``concurrent.futures.ThreadPoolExecutor``. The
# default prefix is ``ThreadPoolExecutor-``; ``token_throttle`` sets explicit
# ``thread_name_prefix="token-throttle-..."`` for its executors. Worker threads
# are daemonic in current CPython, so we must include them explicitly rather
# than relying on a non-daemon filter.
_EXECUTOR_THREAD_NAME_PREFIXES = (
    "ThreadPoolExecutor-",
    "token-throttle-",
)

# Benign runtime/interpreter-internal threads that are not test leaks. The list
# is deliberately small and explicit so an ignore never happens silently:
#   * ``pydevd`` / ``Debugger`` — IDE/debugger helper threads when running under
#     a debugger.
#   * ``asyncio_`` — asyncio's runtime-managed default-executor pool, not an
#     executor the tests own.
_IGNORED_THREAD_NAME_PREFIXES = (
    "pydevd",
    "Debugger",
    "asyncio_",
)

# Snapshot of thread idents present at session start. Stored in a mutable set so
# the start hook can populate it in place without a module-level ``global``
# rebind.
_session_start_thread_ids: set[int] = set()


def _is_executor_worker(thread: threading.Thread) -> bool:
    return (thread.name or "").startswith(_EXECUTOR_THREAD_NAME_PREFIXES)


def _is_ignored_thread(thread: threading.Thread) -> bool:
    return (thread.name or "").startswith(_IGNORED_THREAD_NAME_PREFIXES)


def _is_candidate_leak(thread: threading.Thread) -> bool:
    """True if ``thread`` should be treated as a potential leak.

    A candidate is a live non-daemon thread or a live ThreadPoolExecutor worker,
    excluding explicitly-ignored runtime threads.
    """
    if _is_ignored_thread(thread):
        return False
    if not thread.is_alive():
        return False
    return (not thread.daemon) or _is_executor_worker(thread)


def _describe_thread(thread: threading.Thread) -> str:
    target = getattr(thread, "_target", None)
    target_repr = getattr(target, "__qualname__", None) or repr(target)
    return (
        f"  - name={thread.name!r} daemon={thread.daemon} "
        f"ident={thread.ident} executor_worker={_is_executor_worker(thread)} "
        f"target={target_repr}"
    )


def _lingering_leak_threads() -> list[threading.Thread]:
    """Poll for leaked threads, allowing a grace period for natural shutdown.

    Returns as soon as no candidate leaks remain, or when the grace period
    elapses — whichever comes first. The grace period is what keeps naturally
    finishing executor threads from being flagged as false positives.
    """
    deadline = time.monotonic() + _THREAD_LEAK_GRACE_SECONDS
    while True:
        leaked = [
            thread
            for thread in threading.enumerate()
            if thread.ident not in _session_start_thread_ids
            and _is_candidate_leak(thread)
        ]
        if not leaked or time.monotonic() >= deadline:
            return leaked
        time.sleep(_THREAD_LEAK_POLL_SECONDS)


def pytest_sessionstart(session: pytest.Session) -> None:
    _session_start_thread_ids.clear()
    _session_start_thread_ids.update(thread.ident for thread in threading.enumerate())


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Stay silent on keyboard interrupt / internal error: a half-run session may
    # leave executors mid-shutdown and that is not a real leak signal.
    if exitstatus == pytest.ExitCode.INTERRUPTED:
        return

    leaked = _lingering_leak_threads()
    if not leaked:
        return

    mode = os.environ.get(_THREAD_LEAK_MODE_ENV, "strict").strip().lower()
    report = "\n".join(
        [
            f"Thread-leak detector found {len(leaked)} leaked thread(s) "
            "still alive after the session grace period:",
            *(_describe_thread(thread) for thread in leaked),
        ]
    )

    writer = session.config.pluginmanager.get_plugin("terminalreporter")
    if writer is not None:
        writer.write_line("")
        writer.write_line(report, red=(mode != "report"))

    if mode == "report":
        return

    # STRICT: fail the session. Mutating ``session.exitstatus`` from
    # ``pytest_sessionfinish`` is the supported way to fail without raising
    # (raising here would surface as an internal error instead of a clean fail).
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
