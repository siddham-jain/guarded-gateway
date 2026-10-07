"""lua properties fakeredis can't vouch for (real TIME, NOSCRIPT, memory); run with GG_TEST_REDIS_URL set"""

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from gg.app.redis import connect_redis
from gg.core.keypolicy import RateLimitPolicy
from gg.limits.base import BudgetCaps, BudgetExceededError
from gg.limits.config import LimitsConfig
from gg.limits.ledger import RedisBudgetLedger
from gg.limits.redis_limiter import RedisRateLimiter
from tests.unit.limits.support import fake_clock

pytestmark = pytest.mark.redis

CFG = LimitsConfig()
PREFIX = "ggtest"


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    url = os.environ.get("GG_TEST_REDIS_URL")
    if not url:
        pytest.skip("GG_TEST_REDIS_URL is not set")
    client = connect_redis(url, max_connections=64, pool_timeout_s=0.5)
    await _clear(client)
    yield client
    await _clear(client)
    await client.aclose()


async def _clear(client: Redis) -> None:
    keys = [key async for key in client.scan_iter(f"{PREFIX}:*")]
    if keys:
        await client.delete(*keys)


async def test_500_concurrent_acquires_admit_exactly_the_capacity(redis: Redis) -> None:
    limiter = RedisRateLimiter(redis, CFG.rate_limits, prefix=PREFIX)
    limits = CFG.rate_limits.bucket_limits(RateLimitPolicy(rpm=120, tpm=None, max_concurrent=None))
    results = await asyncio.gather(
        *(limiter.acquire("burst", 0, limits=limits, lease_ttl_s=30) for _ in range(500))
    )
    assert sum(r.allowed for r in results) == 20


async def test_200_concurrent_reservations_admit_exactly_k(redis: Redis) -> None:
    ledger = RedisBudgetLedger(redis, CFG.budgets, clock=fake_clock(), prefix=PREFIX)
    results = await asyncio.gather(
        *(
            ledger.preauthorize("burst", 200, caps=BudgetCaps(daily=200 * 37), hold_ttl_s=60)
            for _ in range(200)
        ),
        return_exceptions=True,
    )
    assert sum(not isinstance(r, BaseException) for r in results) == 37
    assert sum(isinstance(r, BudgetExceededError) for r in results) == 163


async def test_scripts_reload_after_script_flush(redis: Redis) -> None:
    limiter = RedisRateLimiter(redis, CFG.rate_limits, prefix=PREFIX)
    limits = CFG.rate_limits.bucket_limits(RateLimitPolicy(rpm=60, tpm=1_000, max_concurrent=2))
    assert (await limiter.acquire("k", 10, limits=limits, lease_ttl_s=30)).allowed
    await redis.script_flush()
    result = await limiter.acquire("k", 10, limits=limits, lease_ttl_s=30)
    assert result.allowed
    assert result.lease is not None
    await limiter.finish(result.lease, 5)


async def test_memory_per_key_stays_small(redis: Redis) -> None:
    limiter = RedisRateLimiter(redis, CFG.rate_limits, prefix=PREFIX)
    ledger = RedisBudgetLedger(redis, CFG.budgets, clock=fake_clock(), prefix=PREFIX)
    limits = CFG.rate_limits.bucket_limits(RateLimitPolicy(rpm=60, tpm=1_000, max_concurrent=2))
    await limiter.acquire("mem", 10, limits=limits, lease_ttl_s=30)
    await ledger.preauthorize("mem", 100, caps=BudgetCaps(daily=1_000, monthly=5_000), hold_ttl_s=60)
    total = 0
    async for key in redis.scan_iter(f"{PREFIX}:*{{k:mem}}*"):
        total += int(await redis.memory_usage(key) or 0)
    assert total <= 2_048
