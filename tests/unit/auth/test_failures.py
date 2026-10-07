import fakeredis
import pytest

from gg.auth.failures import NAMESPACE, InMemoryAuthFailureLimiter, RedisAuthFailureLimiter
from gg.core.clock import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


async def test_blocks_after_max_failures_until_window_ends(clock: FakeClock) -> None:
    limiter = InMemoryAuthFailureLimiter(clock, max_failures=3, window_s=60)
    for _ in range(2):
        await limiter.record_failure("1.2.3.4")
    assert await limiter.blocked_for("1.2.3.4") is None
    await limiter.record_failure("1.2.3.4")
    assert await limiter.blocked_for("1.2.3.4") == pytest.approx(60)
    assert await limiter.blocked_for("5.6.7.8") is None
    clock.advance(45)
    assert await limiter.blocked_for("1.2.3.4") == pytest.approx(15)
    clock.advance(15)
    assert await limiter.blocked_for("1.2.3.4") is None


async def test_window_resets_the_count(clock: FakeClock) -> None:
    limiter = InMemoryAuthFailureLimiter(clock, max_failures=2, window_s=10)
    await limiter.record_failure("ip")
    clock.advance(11)
    await limiter.record_failure("ip")
    assert await limiter.blocked_for("ip") is None


async def test_redis_count_is_shared_between_limiters(clock: FakeClock) -> None:
    redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    first = RedisAuthFailureLimiter(redis, clock, max_failures=3, window_s=60)
    second = RedisAuthFailureLimiter(redis, clock, max_failures=3, window_s=60)
    await first.record_failure("ip")
    await first.record_failure("ip")
    await second.record_failure("ip")
    assert await second.blocked_for("ip") is not None
    assert await first.blocked_for("ip") is None
    keys = [k.decode() async for k in redis.scan_iter(f"{NAMESPACE}:*")]
    assert len(keys) == 1
    assert 0 < await redis.ttl(keys[0]) <= 60


async def test_redis_errors_fall_back_to_local_counting(clock: FakeClock) -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    limiter = RedisAuthFailureLimiter(
        fakeredis.FakeAsyncRedis(server=server), clock, max_failures=2, window_s=60
    )
    await limiter.record_failure("ip")
    await limiter.record_failure("ip")
    assert await limiter.blocked_for("ip") is not None
