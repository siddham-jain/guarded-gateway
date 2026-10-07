import asyncio
from datetime import UTC, datetime

import pytest

from gg.core.clock import FakeClock
from gg.limits.base import BudgetCaps, BudgetExceededError, BudgetLedger, BudgetUnavailableError
from gg.limits.config import BudgetConfig
from gg.limits.ledger import InMemoryBudgetLedger, RedisBudgetLedger, next_midnight, next_month
from tests.unit.limits.support import RecordingHooks, down_redis, fake_clock, fake_redis

CFG = BudgetConfig()
DAY = "gg:bud:{k:k}:d:20261005"
MONTH = "gg:bud:{k:k}:m:202610"


class Harness:
    """one ledger backend plus a way to read its raw period counters"""

    def __init__(self, kind: str, clock: FakeClock, hooks: RecordingHooks) -> None:
        self.kind = kind
        if kind == "memory":
            self.memory = InMemoryBudgetLedger(CFG, clock=clock, hooks=hooks)
            self.ledger: BudgetLedger = self.memory
        else:
            self.redis = fake_redis()
            self.ledger = RedisBudgetLedger(self.redis, CFG, clock=clock, hooks=hooks, virtual_time=True)

    async def totals(self, period_key: str) -> tuple[int, int]:
        if self.kind == "memory":
            return self.memory.spent(period_key), self.memory.reserved(period_key)
        raw = await self.redis.hmget(period_key, ["spent", "reserved"])
        return int(raw[0] or 0), int(raw[1] or 0)


@pytest.fixture(params=["memory", "redis-lua"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def clock() -> FakeClock:
    return fake_clock()


@pytest.fixture
def hooks() -> RecordingHooks:
    return RecordingHooks()


@pytest.fixture
def harness(kind: str, clock: FakeClock, hooks: RecordingHooks) -> Harness:
    return Harness(kind, clock, hooks)


CAPS = BudgetCaps(daily=1_000, monthly=5_000)


async def test_preauthorize_then_settle_at_actual(harness: Harness) -> None:
    hold, status = await harness.ledger.preauthorize("k", 300, caps=CAPS, hold_ttl_s=60)
    assert hold.period_keys == (DAY, MONTH)
    assert await harness.totals(DAY) == (0, 300)
    # the day is the tightest period: 300/1000 beats 300/5000
    assert (status.period, status.cap, status.used, status.remaining) == ("day", 1_000, 300, 700)
    assert status.resets_at == datetime(2026, 10, 6, tzinfo=UTC)
    await harness.ledger.settle(hold, 120)
    assert hold.state == "settled"
    assert await harness.totals(DAY) == (120, 0)
    assert await harness.totals(MONTH) == (120, 0)
    # settle is idempotent
    await harness.ledger.settle(hold, 120)
    assert await harness.totals(DAY) == (120, 0)


async def test_release_returns_the_reservation(harness: Harness) -> None:
    hold, _ = await harness.ledger.preauthorize("k", 300, caps=CAPS, hold_ttl_s=60)
    await harness.ledger.release(hold)
    assert hold.state == "released"
    assert await harness.totals(DAY) == (0, 0)


async def test_hard_cap_counts_reservations(harness: Harness) -> None:
    first, _ = await harness.ledger.preauthorize("k", 600, caps=CAPS, hold_ttl_s=60)
    with pytest.raises(BudgetExceededError) as info:
        await harness.ledger.preauthorize("k", 401, caps=CAPS, hold_ttl_s=60)
    err = info.value
    assert (err.status, err.code, err.type) == (402, "budget_exceeded", "insufficient_quota")
    assert err.response_headers()["x-should-retry"] == "false"
    assert err.message == (
        "Budget exceeded for key 'k': daily cap $0.001, used $0.0006. Resets at 2026-10-06T00:00:00Z."
    )
    assert (err.budget.period, err.budget.used, err.budget.remaining) == ("day", 600, 400)
    # a refusal reserves nothing; an exact fit still passes
    assert await harness.totals(DAY) == (0, 600)
    second, status = await harness.ledger.preauthorize("k", 400, caps=CAPS, hold_ttl_s=60)
    assert status.ratio == 1.0
    await harness.ledger.settle(first, 100)
    await harness.ledger.settle(second, 100)
    assert await harness.totals(DAY) == (200, 0)


async def test_monthly_cap_refuses_on_its_own(harness: Harness) -> None:
    caps = BudgetCaps(daily=None, monthly=500)
    hold, status = await harness.ledger.preauthorize("k", 500, caps=caps, hold_ttl_s=60)
    assert hold.period_keys == (MONTH,)
    assert (status.period, status.resets_at) == ("month", datetime(2026, 11, 1, tzinfo=UTC))
    with pytest.raises(BudgetExceededError) as info:
        await harness.ledger.preauthorize("k", 1, caps=caps, hold_ttl_s=60)
    assert "monthly cap $0.0005" in info.value.message


async def test_adjust_shrinks_and_grows_with_cap_check(harness: Harness) -> None:
    hold, _ = await harness.ledger.preauthorize("k", 800, caps=CAPS, hold_ttl_s=60)
    assert await harness.ledger.adjust_hold(hold, 300)
    assert hold.amount == 300
    assert await harness.totals(DAY) == (0, 300)
    assert await harness.ledger.extend_hold(hold, 700)
    assert hold.amount == 1_000
    # growing past the cap is refused and leaves the hold unchanged
    assert not await harness.ledger.extend_hold(hold, 1)
    assert hold.amount == 1_000
    assert await harness.totals(DAY) == (0, 1_000)
    await harness.ledger.settle(hold, 900)
    assert not await harness.ledger.adjust_hold(hold, 10)
    assert await harness.totals(DAY) == (900, 0)


async def test_expired_holds_are_swept_as_spent_and_corrected_by_a_late_settle(
    harness: Harness, clock: FakeClock, hooks: RecordingHooks
) -> None:
    stale, _ = await harness.ledger.preauthorize("k", 400, caps=CAPS, hold_ttl_s=10)
    clock.advance(11)
    fresh, _ = await harness.ledger.preauthorize("k", 100, caps=CAPS, hold_ttl_s=10)
    assert hooks.expired == 1
    assert await harness.totals(DAY) == (400, 100)
    # the slow request finally settles: swept 400 becomes the true 150
    await harness.ledger.settle(stale, 150)
    await harness.ledger.settle(fresh, 50)
    assert await harness.totals(DAY) == (200, 0)


async def test_utc_midnight_rollover(harness: Harness, clock: FakeClock) -> None:
    caps = BudgetCaps(daily=1_000, monthly=5_000)
    late, _ = await harness.ledger.preauthorize("k", 900, caps=caps, hold_ttl_s=7_200)
    clock.advance(3_600)
    # a new utc day starts clean; the month keeps counting
    hold, status = await harness.ledger.preauthorize("k", 900, caps=caps, hold_ttl_s=60)
    assert hold.period_keys == ("gg:bud:{k:k}:d:20261006", MONTH)
    assert status.used == 900
    # the earlier hold settles against the day it reserved in
    await harness.ledger.settle(late, 800)
    assert await harness.totals(DAY) == (800, 0)
    assert await harness.totals("gg:bud:{k:k}:d:20261006") == (0, 900)
    assert await harness.totals(MONTH) == (800, 900)


async def test_concurrent_reservations_admit_exactly_what_fits(harness: Harness) -> None:
    caps = BudgetCaps(daily=50 * 200)
    results = await asyncio.gather(
        *(harness.ledger.preauthorize("k", 200, caps=caps, hold_ttl_s=60) for _ in range(200)),
        return_exceptions=True,
    )
    granted = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BudgetExceededError)]
    assert (len(granted), len(refused)) == (50, 150)
    assert await harness.totals(DAY) == (0, 10_000)


async def test_negative_amounts_are_rejected(harness: Harness) -> None:
    with pytest.raises(ValueError, match="negative"):
        await harness.ledger.preauthorize("k", -1, caps=CAPS, hold_ttl_s=60)


async def test_budget_keys_never_expire(clock: FakeClock) -> None:
    redis = fake_redis()
    ledger = RedisBudgetLedger(redis, CFG, clock=clock)
    hold, _ = await ledger.preauthorize("k", 100, caps=CAPS, hold_ttl_s=60)
    for key in (DAY, MONTH, "gg:bud:{k:k}:holds", f"gg:bud:{{k:k}}:h:{hold.hold_id}"):
        assert await redis.ttl(key) == -1, key
    await ledger.settle(hold, 50)
    # only the settled tombstone is evictable
    assert await redis.ttl(f"gg:bud:{{k:k}}:h:{hold.hold_id}") > 0
    assert await redis.ttl(DAY) == -1


async def test_redis_outage_fails_closed(clock: FakeClock) -> None:
    hooks = RecordingHooks()
    ledger = RedisBudgetLedger(down_redis(), CFG, clock=clock, hooks=hooks)
    with pytest.raises(BudgetUnavailableError) as info:
        await ledger.preauthorize("k", 100, caps=CAPS, hold_ttl_s=60)
    assert (info.value.status, info.value.code) == (503, "budget_unavailable")
    assert info.value.response_headers()["retry-after"] == "5"
    assert hooks.errors == ["reserve"]


async def test_settle_during_outage_keeps_the_hold_open(clock: FakeClock) -> None:
    hooks = RecordingHooks()
    healthy = fake_redis()
    ledger = RedisBudgetLedger(healthy, CFG, clock=clock, hooks=hooks)
    hold, _ = await ledger.preauthorize("k", 100, caps=CAPS, hold_ttl_s=60)
    broken = RedisBudgetLedger(down_redis(), CFG, clock=clock, hooks=hooks)
    await broken.settle(hold, 10)
    assert hold.state == "open"
    assert not await broken.adjust_hold(hold, 50)
    assert hooks.errors == ["settle", "adjust"]


def test_period_boundaries() -> None:
    assert next_midnight(datetime(2026, 12, 31, 23, 59, tzinfo=UTC)) == datetime(2027, 1, 1, tzinfo=UTC)
    assert next_month(datetime(2026, 12, 15, tzinfo=UTC)) == datetime(2027, 1, 1, tzinfo=UTC)
    assert next_month(datetime(2026, 2, 28, tzinfo=UTC)) == datetime(2026, 3, 1, tzinfo=UTC)
