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
    config: pytest.Config,  # noqa: ARG001
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
