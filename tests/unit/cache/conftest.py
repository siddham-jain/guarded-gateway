import os

import pytest

REDIS_URL_ENV = "GG_TEST_REDIS_URL"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # real redis:8 tests (FT.* is missing from fakeredis) only run when a server is named
    if os.environ.get(REDIS_URL_ENV):
        return
    skip = pytest.mark.skip(reason=f"needs a real redis 8; set {REDIS_URL_ENV}")
    for item in items:
        if item.get_closest_marker("redis") and item.path.is_relative_to(os.path.dirname(__file__)):
            item.add_marker(skip)
