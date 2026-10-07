import asyncio
from collections.abc import Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gg.core.clock import FakeClock
from gg.core.keypolicy import RateLimitPolicy
from gg.limits.base import BucketLimits, Lease, RateLimiter
from gg.limits.bucket import BucketLevel, LocalRateLimiter, reconcile, refill, reset_ms, wait_ms
from gg.limits.config import RateLimitConfig
from gg.limits.redis_limiter import RedisRateLimiter
from tests.unit.limits.support import fake_clock, fake_redis

CFG = RateLimitConfig()
TTL_S = 60.0

type Factory = Callable[[FakeClock], RateLimiter]


def local(clock: FakeClock) -> RateLimiter:
    return LocalRateLimiter(clock)


def redis_backed(clock: FakeClock) -> RateLimiter:
    # virtual time drives the lua scripts so both backends see the same clock
    return RedisRateLimiter(fake_redis(), CFG, now_ms=lambda: int(clock.time() * 1000))


BACKENDS = [pytest.param(local, id="local"), pytest.param(redis_backed, id="redis-lua")]


def limits(rpm: int | None = None, tpm: int | None = None, conc: int | None = None) -> BucketLimits:
    return CFG.bucket_limits(RateLimitPolicy(rpm=rpm, tpm=tpm, max_concurrent=conc))


def test_bucket_capacity_from_burst_windows() -> None:
    lim = limits(rpm=60, tpm=100_000, conc=5)
    assert (lim.rpm_capacity, lim.tpm_capacity, lim.max_concurrent) == (10.0, 100_000.0, 5)
    # tiny rpm still admits one request
    assert limits(rpm=3).rpm_capacity == 1.0
    assert not limits().enabled


def test_bucket_math_by_hand() -> None:
    # empty state starts full; 60 rpm refills one token per second
    assert refill(None, 5_000, 60, 10.0) == 10.0
    assert refill(BucketLevel(0.0, 1_000), 1_500, 60, 10.0) == 0.5
    assert refill(BucketLevel(9.5, 1_000), 10_000, 60, 10.0) == 10.0
    # time going backwards never drains the bucket
    assert refill(BucketLevel(3.0, 2_000), 1_000, 60, 10.0) == 3.0
    assert wait_ms(0.5, 1, 60) == 500
    assert wait_ms(2.0, 1, 60) == 0
    assert reset_ms(7.0, 60, 10.0) == 3_000
    assert reconcile(100.0, -500, 300.0) == 300.0
    assert reconcile(100.0, 1_000, 300.0) == -300.0


@pytest.mark.parametrize("factory", BACKENDS)
async def test_rpm_burst_then_refill(factory: Factory) -> None:
    clock = fake_clock()
    limiter = factory(clock)
    lim = limits(rpm=60)
    results = [await limiter.acquire("k", 0, limits=lim, lease_ttl_s=TTL_S) for _ in range(11)]
    assert [r.allowed for r in results] == [True] * 10 + [False]
    assert [r.remaining_requests for r in results[:3]] == [9, 8, 7]
    assert results[0].reset_requests_s == 1.0
    denied = results[-1]
    assert denied.reason == "rpm"
    assert denied.retry_after_s == 1.0
    assert denied.lease is None
    clock.advance(0.999)
    assert not (await limiter.acquire("k", 0, limits=lim, lease_ttl_s=TTL_S)).allowed
    clock.advance(0.001)
    assert (await limiter.acquire("k", 0, limits=lim, lease_ttl_s=TTL_S)).allowed


@pytest.mark.parametrize("factory", BACKENDS)
async def test_keys_are_isolated(factory: Factory) -> None:
    limiter = factory(fake_clock())
    lim = limits(rpm=6)
    assert (await limiter.acquire("a", 0, limits=lim, lease_ttl_s=TTL_S)).allowed
    assert not (await limiter.acquire("a", 0, limits=lim, lease_ttl_s=TTL_S)).allowed
    assert (await limiter.acquire("b", 0, limits=lim, lease_ttl_s=TTL_S)).allowed


@pytest.mark.parametrize("factory", BACKENDS)
async def test_tpm_bucket_and_retry_after(factory: Factory) -> None:
    clock = fake_clock()
    limiter = factory(clock)
    lim = limits(tpm=6_000)
    first = await limiter.acquire("k", 4_000, limits=lim, lease_ttl_s=TTL_S)
    assert first.allowed
    assert first.remaining_tokens == 2_000
    assert first.reset_tokens_s == 40.0
    assert first.remaining_requests is None
    second = await limiter.acquire("k", 3_000, limits=lim, lease_ttl_s=TTL_S)
    # 1000 tokens short at 100 tokens/s
    assert (second.allowed, second.reason, second.retry_after_s) == (False, "tpm", 10.0)
    clock.advance(10)
    assert (await limiter.acquire("k", 3_000, limits=lim, lease_ttl_s=TTL_S)).allowed


@pytest.mark.parametrize("factory", BACKENDS)
async def test_need_above_capacity_can_never_pass(factory: Factory) -> None:
    limiter = factory(fake_clock())
    result = await limiter.acquire("k", 6_001, limits=limits(rpm=60, tpm=6_000), lease_ttl_s=TTL_S)
    assert (result.allowed, result.reason, result.retry_after_s) == (False, "tokens_exceed_limit", None)
    # a rejection consumes nothing
    ok = await limiter.acquire("k", 6_000, limits=limits(rpm=60, tpm=6_000), lease_ttl_s=TTL_S)
    assert ok.allowed
    assert ok.remaining_requests == 9


@pytest.mark.parametrize("factory", BACKENDS)
async def test_rejection_on_one_limit_consumes_neither(factory: Factory) -> None:
    clock = fake_clock()
    limiter = factory(clock)
    lim = limits(rpm=60, tpm=1_000)
    assert (await limiter.acquire("k", 900, limits=lim, lease_ttl_s=TTL_S)).allowed
    denied = await limiter.acquire("k", 900, limits=lim, lease_ttl_s=TTL_S)
    assert denied.reason == "tpm"
    assert denied.remaining_requests == 9


@pytest.mark.parametrize("factory", BACKENDS)
async def test_concurrency_leases_release_and_expire(factory: Factory) -> None:
    clock = fake_clock()
    limiter = factory(clock)
    lim = limits(conc=2)
    a = await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)
    b = await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)
    c = await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)
    assert (a.allowed, b.allowed, c.allowed) == (True, True, False)
    assert (c.reason, c.retry_after_s) == ("concurrency", 1.0)
    assert a.lease is not None
    await limiter.finish(a.lease, None)
    assert (await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)).allowed
    assert not (await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)).allowed
    # a crashed request (no finish) frees its slot once the lease ttl passes
    clock.advance(5)
    assert (await limiter.acquire("k", 0, limits=lim, lease_ttl_s=5)).allowed


@pytest.mark.parametrize("factory", BACKENDS)
async def test_tpm_reconcile_refund_and_debit(factory: Factory) -> None:
    clock = fake_clock()
    limiter = factory(clock)
    lim = limits(tpm=6_000)
    first = await limiter.acquire("k", 5_000, limits=lim, lease_ttl_s=TTL_S)
    assert first.lease is not None
    # used 1_000 of 5_000: 4_000 come back
    await limiter.finish(first.lease, 1_000)
    probe = await limiter.acquire("k", 5_000, limits=lim, lease_ttl_s=TTL_S)
    assert probe.allowed
    assert probe.remaining_tokens == 0
    assert probe.lease is not None
    # used 20_000 against an estimate of 5_000: debt floors at -capacity
    await limiter.finish(probe.lease, 20_000)
    denied = await limiter.acquire("k", 1, limits=lim, lease_ttl_s=TTL_S)
    assert (denied.allowed, denied.remaining_tokens) == (False, 0)
    assert denied.retry_after_s == 60.01
    clock.advance(60.01)
    assert (await limiter.acquire("k", 1, limits=lim, lease_ttl_s=TTL_S)).allowed


@pytest.mark.parametrize("factory", BACKENDS)
async def test_refund_caps_at_capacity(factory: Factory) -> None:
    limiter = factory(fake_clock())
    lim = limits(tpm=1_000)
    result = await limiter.acquire("k", 100, limits=lim, lease_ttl_s=TTL_S)
    assert result.lease is not None
    await limiter.finish(result.lease, 0)
    again = await limiter.acquire("k", 0, limits=lim, lease_ttl_s=TTL_S)
    assert again.remaining_tokens == 1_000


@pytest.mark.parametrize("factory", BACKENDS)
async def test_no_limits_is_a_free_pass(factory: Factory) -> None:
    limiter = factory(fake_clock())
    result = await limiter.acquire("k", 10**9, limits=limits(), lease_ttl_s=TTL_S)
    assert result.allowed
    assert result.lease is None
    assert result.remaining_requests is None


async def test_redis_keys_are_hash_tagged_with_ttls() -> None:
    redis = fake_redis()
    limiter = RedisRateLimiter(redis, CFG)
    await limiter.acquire("demo", 10, limits=limits(rpm=60, tpm=1_000, conc=2), lease_ttl_s=TTL_S)
    keys = sorted(k.decode() for k in await redis.keys("*"))
    assert keys == ["gg:rl:{k:demo}:conc", "gg:rl:{k:demo}:rpm", "gg:rl:{k:demo}:tpm"]
    for key in keys:
        assert await redis.pttl(key) > 0


async def test_production_limiter_never_overrides_redis_time(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = RedisRateLimiter(fake_redis(), CFG)
    seen: list[list[object]] = []
    admit, finish = limiter._admit, limiter._finish

    async def spy_admit(*, keys: list[str], args: list[object]) -> object:
        seen.append(args)
        return await admit(keys=keys, args=args)

    async def spy_finish(*, keys: list[str], args: list[object]) -> object:
        seen.append(args)
        return await finish(keys=keys, args=args)

    monkeypatch.setattr(limiter, "_admit", spy_admit)
    monkeypatch.setattr(limiter, "_finish", spy_finish)
    result = await limiter.acquire("k", 10, limits=limits(rpm=60, tpm=1_000, conc=1), lease_ttl_s=TTL_S)
    assert result.lease is not None
    await limiter.finish(result.lease, 5)
    assert [args[-1] for args in seen] == ["", ""]


async def test_lua_admit_is_atomic_under_concurrency() -> None:
    # 200 concurrent acquires against a bucket of 50 admit exactly 50
    limiter = RedisRateLimiter(fake_redis(), CFG)
    lim = limits(rpm=300)
    results = await asyncio.gather(
        *(limiter.acquire("burst", 0, limits=lim, lease_ttl_s=TTL_S) for _ in range(200))
    )
    assert sum(r.allowed for r in results) == 50


async def test_lua_concurrency_is_atomic_under_concurrency() -> None:
    limiter = RedisRateLimiter(fake_redis(), CFG)
    lim = limits(conc=50)
    results = await asyncio.gather(
        *(limiter.acquire("burst", 0, limits=lim, lease_ttl_s=TTL_S) for _ in range(200))
    )
    assert sum(r.allowed for r in results) == 50
    assert {r.reason for r in results if not r.allowed} == {"concurrency"}


steps = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=5_000),
        st.integers(min_value=0, max_value=2_500),
        st.sampled_from(["acquire", "finish"]),
        st.integers(min_value=0, max_value=3_000),
    ),
    min_size=1,
    max_size=40,
)


@settings(max_examples=60, deadline=None)
@given(steps=steps, rpm=st.integers(1, 600), tpm=st.integers(1, 5_000), conc=st.integers(0, 4))
async def test_lua_matches_python_reference(
    steps: list[tuple[int, int, str, int]], rpm: int, tpm: int, conc: int
) -> None:
    clock = fake_clock()
    reference = LocalRateLimiter(clock)
    lua = redis_backed(clock)
    lim = limits(rpm=rpm, tpm=tpm, conc=conc or None)
    leases: list[tuple[Lease, Lease]] = []
    for advance_ms, need, op, actual in steps:
        clock.advance(advance_ms / 1000)
        if op == "finish" and leases:
            ref_lease, lua_lease = leases.pop(0)
            await reference.finish(ref_lease, actual)
            await lua.finish(lua_lease, actual)
            continue
        want = await reference.acquire("k", need, limits=lim, lease_ttl_s=2)
        got = await lua.acquire("k", need, limits=lim, lease_ttl_s=2)
        assert (got.allowed, got.reason, got.retry_after_s) == (want.allowed, want.reason, want.retry_after_s)
        assert (got.remaining_requests, got.remaining_tokens) == (
            want.remaining_requests,
            want.remaining_tokens,
        )
        assert (got.reset_requests_s, got.reset_tokens_s) == (want.reset_requests_s, want.reset_tokens_s)
        if want.lease is not None and got.lease is not None:
            leases.append((want.lease, got.lease))
