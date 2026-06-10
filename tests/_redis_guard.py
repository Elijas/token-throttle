"""Session safety gate around Redis ``flushdb`` in the test suite.

The integration / property / Redis-touching unit tests flush the configured
Redis database *around every test* to isolate state. That is safe against a
dedicated empty test database, but catastrophic if a contributor points
``--redis-url`` at a shared or production host: the suite would silently wipe it.

This module installs a one-time-per-session gate, keyed by Redis URL. The first
time any flush call site is reached for a given URL, :func:`ensure_flush_allowed`
checks ``DBSIZE``:

* If the database is empty (``DBSIZE == 0``) it proceeds. From then on only
  test keys can exist, so within-session flushes are safe and the check is not
  repeated for that URL.
* If the database is non-empty and ``TOKEN_THROTTLE_TESTS_ALLOW_FLUSH=1`` is
  set, it proceeds (the contributor has explicitly accepted data loss).
* Otherwise it aborts the whole pytest session with an actionable message,
  before any ``flushdb`` runs.

The gate makes its *own* short-lived synchronous connection for the ``DBSIZE``
probe, so a single helper covers both the async and sync client paths. If the
probe cannot connect (server down), the helper does nothing: the test's own
``ping`` / ``importorskip`` path is left to skip the test exactly as it does
today. The gate only ever fires when a connection is actually established.
"""

from __future__ import annotations

import os
import threading

import pytest

_ALLOW_FLUSH_ENV = "TOKEN_THROTTLE_TESTS_ALLOW_FLUSH"

# URLs already cleared by the gate this session. A URL is recorded only after a
# successful, permitted DBSIZE check, so a flaky/non-empty DB is never silently
# remembered as "ok".
_cleared_urls: set[str] = set()
_lock = threading.Lock()


def _abort_message(redis_url: str, dbsize: int) -> str:
    return (
        f"\n\nRefusing to run: the test suite flushes the Redis database around "
        f"every test, but {redis_url!r} is NOT empty (DBSIZE={dbsize}).\n"
        f"Running here would DELETE that data.\n\n"
        f"Fix one of:\n"
        f"  * Point --redis-url at a dedicated, empty test database. The safest "
        f"option is a spare DB index on the same server, e.g.\n"
        f"        --redis-url redis://localhost:6379/13\n"
        f"  * If you genuinely want to flush THIS database, accept the data loss "
        f"explicitly:\n"
        f"        {_ALLOW_FLUSH_ENV}=1 pytest ...\n"
    )


def ensure_flush_allowed(redis_url: str) -> None:
    """Gate ``flushdb`` for ``redis_url``; abort the session if unsafe.

    Idempotent per session and per URL: the ``DBSIZE`` probe runs at most once
    per URL. Safe to call at every flush site, on both the sync and async paths.
    """
    with _lock:
        if redis_url in _cleared_urls:
            return

        if os.environ.get(_ALLOW_FLUSH_ENV) == "1":
            # Caller has explicitly accepted data loss for this run.
            _cleared_urls.add(redis_url)
            return

        # Probe DBSIZE with a one-shot synchronous client. This works whether the
        # caller is on the sync or async path (no event loop required here).
        try:
            import redis as sync_redis  # noqa: PLC0415
            from redis.exceptions import RedisError  # noqa: PLC0415
        except ImportError:
            # No redis package installed: the caller's own importorskip/ping will
            # skip the test. Nothing to gate.
            return

        client = sync_redis.from_url(redis_url)
        try:
            dbsize = client.dbsize()
        except RedisError:
            # Server unreachable (or auth/cluster error). Do NOT abort here: the
            # caller's ping/importorskip path skips the test as it does today.
            # The gate only fires on an established connection.
            return
        finally:
            client.close()

        if dbsize != 0:
            pytest.exit(_abort_message(redis_url, dbsize), returncode=1)

        # Database is empty: from now on only test keys exist, so flushes are
        # safe and we don't re-probe this URL.
        _cleared_urls.add(redis_url)
