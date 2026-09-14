"""Route the differential harness's Redis URL through the suite's ``--redis-url`` option."""

from __future__ import annotations

import pytest

from tests.differential import _backends


@pytest.fixture(autouse=True, scope="session")
def _differential_redis_url(request: pytest.FixtureRequest) -> None:
    _backends.set_pytest_redis_url(str(request.config.getoption("--redis-url")))
