import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

from frozendict import frozendict

from token_throttle._factories._openai._token_counter import OpenAIUsageCounter
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._rate_limiter import RateLimiter


class _FakeEncoding:
    def encode(self, text: str, **_kwargs: object) -> list[int]:
        return list(range(len(text)))


def _backend_builder():
    backend = AsyncMock()
    backend.await_for_capacity.return_value = None
    builder = MagicMock()
    builder.build.return_value = backend
    return builder


def _config(counter: OpenAIUsageCounter) -> PerModelConfig:
    return PerModelConfig(
        quotas=UsageQuotas(
            [
                Quota(metric="tokens", limit=10_000),
                Quota(metric="requests", limit=10_000),
            ],
        ),
        usage_counter=counter,
        model_family="gpt-4o",
    )


async def test_async_openai_counter_cold_load_does_not_block_event_loop():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = threading.Event()
    loop_progressed_while_loading = asyncio.Event()

    def blocking_get_encoding(_model: str) -> _FakeEncoding:
        loop.call_soon_threadsafe(started.set)
        time.sleep(0.05)
        finished.set()
        return _FakeEncoding()

    counter = OpenAIUsageCounter(get_encoding_func=blocking_get_encoding)
    limiter = RateLimiter(_config(counter), backend=_backend_builder())

    async def observe_loop_progress() -> None:
        await started.wait()
        await asyncio.sleep(0)
        if not finished.is_set():
            loop_progressed_while_loading.set()

    await asyncio.gather(
        limiter.acquire_capacity_for_request(model="gpt-4o", input="hello"),
        observe_loop_progress(),
    )

    assert loop_progressed_while_loading.is_set()


def test_sync_openai_counter_loads_encoding_on_calling_thread():
    calling_thread = threading.get_ident()
    loader_thread: int | None = None

    def get_encoding(_model: str) -> _FakeEncoding:
        nonlocal loader_thread
        loader_thread = threading.get_ident()
        return _FakeEncoding()

    counter = OpenAIUsageCounter(get_encoding_func=get_encoding)

    assert counter("gpt-4o", input="hello") == frozendict({"tokens": 5, "requests": 1})
    assert loader_thread == calling_thread


async def test_warmup_models_preloads_multiple_models_for_later_sync_use():
    calls: list[str] = []

    def get_encoding(model: str) -> _FakeEncoding:
        calls.append(model)
        return _FakeEncoding()

    counter = OpenAIUsageCounter(get_encoding_func=get_encoding)

    await counter.warmup_models(["gpt-4o", "gpt-3.5-turbo"])

    assert sorted(calls) == ["gpt-3.5-turbo", "gpt-4o"]
    calls.clear()

    assert await counter.count_request_async(
        model="gpt-4o", input="hello"
    ) == frozendict({"tokens": 5, "requests": 1})
    assert counter("gpt-3.5-turbo", input="hi") == frozendict(
        {"tokens": 2, "requests": 1}
    )
    assert calls == []
