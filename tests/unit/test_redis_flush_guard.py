"""Unit tests for the Redis flush safety gate (`tests/_redis_guard.py`).

These exercise the gate's decision logic by faking a synchronous redis client,
so they never connect to or flush a real Redis database.
"""

from __future__ import annotations

import sys
import types

import pytest

import tests._redis_guard as guard
from tests._redis_guard import _ALLOW_FLUSH_ENV, ensure_flush_allowed


class _FakeRedisError(Exception):
    pass


class _FakeClient:
    def __init__(self, *, dbsize: int | None = None, raise_on_dbsize: bool = False):
        self._dbsize = dbsize
        self._raise = raise_on_dbsize
        self.closed = False

    def dbsize(self) -> int:
        if self._raise:
            raise _FakeRedisError("connection refused")
        assert self._dbsize is not None
        return self._dbsize

    def close(self) -> None:
        self.closed = True


def _install_fake_redis(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    """Make ``import redis`` / ``from redis.exceptions import RedisError`` inside
    the guard resolve to our fake, recording the URL the guard connects to.
    """
    fake_redis = types.ModuleType("redis")
    fake_redis.from_url = lambda url: setattr(client, "url", url) or client  # type: ignore[attr-defined]
    fake_exceptions = types.ModuleType("redis.exceptions")
    fake_exceptions.RedisError = _FakeRedisError  # type: ignore[attr-defined]
    fake_redis.exceptions = fake_exceptions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setitem(sys.modules, "redis.exceptions", fake_exceptions)


@pytest.fixture(autouse=True)
def _reset_gate_cache(monkeypatch: pytest.MonkeyPatch):
    """Each test gets a fresh per-session cache and no opt-out env var."""
    monkeypatch.setattr(guard, "_cleared_urls", set())
    monkeypatch.delenv(_ALLOW_FLUSH_ENV, raising=False)


def test_nonempty_db_aborts_session(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(dbsize=7)
    _install_fake_redis(monkeypatch, client)

    with pytest.raises(pytest.exit.Exception) as excinfo:
        ensure_flush_allowed("redis://example/0")

    msg = str(excinfo.value)
    assert "flushes the Redis database" in msg
    assert "DBSIZE=7" in msg
    assert _ALLOW_FLUSH_ENV in msg
    assert client.closed is True


def test_empty_db_proceeds_and_caches(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(dbsize=0)
    _install_fake_redis(monkeypatch, client)

    ensure_flush_allowed("redis://example/0")  # no raise

    assert "redis://example/0" in guard._cleared_urls
    assert client.closed is True


def test_env_var_opt_out_skips_probe(monkeypatch: pytest.MonkeyPatch):
    # dbsize would be non-empty, but the opt-out env var must short-circuit
    # before any probe connection is made.
    client = _FakeClient(dbsize=999)
    _install_fake_redis(monkeypatch, client)
    monkeypatch.setenv(_ALLOW_FLUSH_ENV, "1")

    ensure_flush_allowed("redis://example/0")  # no raise

    assert "redis://example/0" in guard._cleared_urls
    # Probe was never opened, so the fake was never closed.
    assert client.closed is False


def test_connection_error_does_not_abort(monkeypatch: pytest.MonkeyPatch):
    # Server unreachable: the gate must stay silent (caller's ping/skip handles
    # it) and must NOT cache the URL as cleared.
    client = _FakeClient(raise_on_dbsize=True)
    _install_fake_redis(monkeypatch, client)

    ensure_flush_allowed("redis://down/0")  # no raise

    assert "redis://down/0" not in guard._cleared_urls
    assert client.closed is True


def test_second_call_is_cached(monkeypatch: pytest.MonkeyPatch):
    client = _FakeClient(dbsize=0)
    _install_fake_redis(monkeypatch, client)

    ensure_flush_allowed("redis://example/0")
    assert client.closed is True

    # A second call must not re-probe (swap in a client that would abort if used).
    poison = _FakeClient(dbsize=12345)
    _install_fake_redis(monkeypatch, poison)
    ensure_flush_allowed("redis://example/0")  # cached: no raise, no probe
    assert poison.closed is False
