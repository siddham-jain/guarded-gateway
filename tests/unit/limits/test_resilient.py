from dataclasses import replace

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from gg.core.clock import FakeClock
from gg.core.keypolicy import RateLimitPolicy
from gg.limits.base import BucketLimits, Lease, LimitResult, RateLimiterUnavailableError
from gg.limits.bucket import LocalRateLimiter
from gg.limits.config import RateLimitConfig
from gg.limits.redis_limiter import RedisRateLimiter, ResilientRateLimiter
from tests.unit.limits.support import RecordingHooks, down_redis, fake_clock, fake_redis

CFG = RateLimitConfig()
LIMITS = CFG.bucket_limits(RateLimitPolicy(rpm=60, tpm=None, max_concurrent=None))
CONC = CFG.bucket_limits(RateLimitPolicy(rpm=None, tpm=None, max_concurrent=1))


class FlakyLimiter:
    """a primary that fails while `down` is set and counts the calls it receives"""

    def __init__(self, clock: FakeClock) -> None:
        self.down = True
        self.calls = 0
        self.finished: list[Lease] = []
        self._inner = LocalRateLimiter(clock)

    async def acquire(
        self, key_id: str, tokens_estimate: int, /, *, limits: BucketLimits, lease_ttl_s: float
    ) -> LimitResult:
        self.calls += 1
        if self.down:
            raise RedisConnectionError("redis is down")
        result = await self._inner.acquire(key_id, tokens_estimate, limits=limits, lease_ttl_s=lease_ttl_s)
        lease = replace(result.lease, local=False) if result.lease is not None else None
        return replace(result, lease=lease)

    async def finish(self, lease: Lease, actual_tokens: int | None, /) -> None:
        if self.down:
            raise RedisConnectionError("redis is down")
        self.finished.append(lease)


async def test_outage_fails_open_to_local_bucket_and_breaker_skips_redis() -> None:
    clock = fake_clock()
    primary = FlakyLimiter(clock)
    hooks = RecordingHooks()
    limiter = ResilientRateLimiter(primary, LocalRateLimiter(clock), CFG, clock=clock, hooks=hooks)
    results = [await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30) for _ in range(11)]
    # the local bucket still enforces the limit while degraded
    assert [r.allowed for r in results] == [True] * 10 + [False]
    assert all(r.degraded for r in results)
    assert all(r.lease is None or r.lease.local for r in results)
    # three failures open the breaker; later calls skip redis entirely
    assert primary.calls == 3
    assert hooks.errors == ["acquire"] * 3
    assert hooks.degraded_flags == [True]
    assert limiter.degraded

    # a failed half-open probe reopens the breaker straight away
    clock.advance(CFG.breaker_cooldown_s)
    await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    assert primary.calls == 4

    primary.down = False
    clock.advance(CFG.breaker_cooldown_s)
    recovered = await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    assert recovered.allowed
    assert not recovered.degraded
    assert hooks.degraded_flags == [True, False]
    assert not limiter.degraded


async def test_finish_routes_to_the_backend_that_issued_the_lease() -> None:
    clock = fake_clock()
    primary = FlakyLimiter(clock)
    fallback = LocalRateLimiter(clock)
    limiter = ResilientRateLimiter(primary, fallback, CFG, clock=clock)
    local = await limiter.acquire("k", 0, limits=CONC, lease_ttl_s=30)
    assert local.lease is not None
    assert local.lease.local
    assert fallback.in_use("k") == 1
    await limiter.finish(local.lease, 0)
    assert fallback.in_use("k") == 0

    primary.down = False
    clock.advance(CFG.breaker_cooldown_s)
    remote = await limiter.acquire("k", 0, limits=CONC, lease_ttl_s=30)
    assert remote.lease is not None
    assert not remote.lease.local
    await limiter.finish(remote.lease, 0)
    assert primary.finished == [remote.lease]


async def test_finish_failure_is_swallowed_and_counted() -> None:
    clock = fake_clock()
    primary = FlakyLimiter(clock)
    primary.down = False
    hooks = RecordingHooks()
    limiter = ResilientRateLimiter(primary, LocalRateLimiter(clock), CFG, clock=clock, hooks=hooks)
    result = await limiter.acquire("k", 0, limits=CONC, lease_ttl_s=30)
    assert result.lease is not None
    primary.down = True
    await limiter.finish(result.lease, 0)
    assert hooks.errors == ["finish"]


async def test_fail_closed_mode_rejects_with_503() -> None:
    clock = fake_clock()
    cfg = RateLimitConfig(fail_mode="closed")
    limiter = ResilientRateLimiter(FlakyLimiter(clock), LocalRateLimiter(clock), cfg, clock=clock)
    with pytest.raises(RateLimiterUnavailableError) as info:
        await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    assert info.value.status == 503
    assert info.value.code == "ratelimit_unavailable"


async def test_real_redis_client_outage_falls_back() -> None:
    clock = fake_clock()
    limiter = ResilientRateLimiter(
        RedisRateLimiter(down_redis(), CFG), LocalRateLimiter(clock), CFG, clock=clock
    )
    result = await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    assert (result.allowed, result.degraded) == (True, True)


async def test_healthy_redis_is_not_degraded() -> None:
    clock = fake_clock()
    limiter = ResilientRateLimiter(
        RedisRateLimiter(fake_redis(), CFG), LocalRateLimiter(clock), CFG, clock=clock
    )
    result = await limiter.acquire("k", 0, limits=LIMITS, lease_ttl_s=30)
    assert (result.allowed, result.degraded) == (True, False)
    assert result.lease is None
