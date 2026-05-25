from token_throttle import (
    BackendIntrospectable,
    BackendIntrospectionDiagnostic,
    SyncBackendIntrospectable,
)
from token_throttle._interfaces._interfaces import PerModelConfig
from token_throttle._interfaces._models import Quota, UsageQuotas
from token_throttle._limiter_backends._memory._backend import MemoryBackendBuilder
from token_throttle._limiter_backends._memory._sync_backend import (
    SyncMemoryBackendBuilder,
)


def _config() -> PerModelConfig:
    return PerModelConfig(
        model_family="protocol-family",
        quotas=UsageQuotas([Quota(metric="tokens", limit=100, per_seconds=60)]),
    )


async def test_async_memory_backend_conforms_to_introspection_protocol():
    backend = MemoryBackendBuilder().build(_config())

    assert isinstance(backend, BackendIntrospectable)
    diagnostic = await backend.introspect()

    assert BackendIntrospectionDiagnostic.model_validate(diagnostic) == diagnostic
    assert diagnostic.backend_type == "memory"
    assert diagnostic.model_family == "protocol-family"
    assert diagnostic.buckets[0].metric == "tokens"
    assert diagnostic.memory_health is not None


def test_sync_memory_backend_conforms_to_introspection_protocol():
    backend = SyncMemoryBackendBuilder().build(_config())

    assert isinstance(backend, SyncBackendIntrospectable)
    diagnostic = backend.introspect()

    assert BackendIntrospectionDiagnostic.model_validate(diagnostic) == diagnostic
    assert diagnostic.backend_type == "memory"
    assert diagnostic.model_family == "protocol-family"
    assert diagnostic.buckets[0].metric == "tokens"
    assert diagnostic.memory_health is not None
