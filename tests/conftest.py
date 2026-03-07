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
    """Auto-mark integration tests with the 'redis' marker."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.redis)
