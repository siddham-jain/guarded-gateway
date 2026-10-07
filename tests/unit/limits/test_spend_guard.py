from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import fakeredis
import pytest
from pydantic import ValidationError

from gg.core.clock import FakeClock
from gg.limits.base import ProviderSpendGuard, SpendCapReachedError, SpendDenial
from gg.limits.spend_guard import InMemorySpendGuard, RedisSpendGuard, SpendGuardSettings

DAY = 86_400


class Telemetry:
    def __init__(self) -> None:
        self.denied: list[tuple[str, str]] = []

    def pricing_missing(self, provider: str, deployment: str, /) -> None:
        raise AssertionError("not expected")

    def spend_denied(self, provider: str, reason: str, /) -> None:
        self.denied.append((provider, reason))


def settings(**overrides: Any) -> SpendGuardSettings:
    return SpendGuardSettings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def clock() -> FakeClock:
    # 2026-10-05 23:00 utc, one hour before the day rolls over
    return FakeClock(wall=datetime(2026, 10, 5, 23, 0, tzinfo=UTC).timestamp())


def test_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GG_SPEND_CAP_DAILY_USD", "openai=0.50, gemini = 0.25")
    monkeypatch.setenv("GG_SPEND_CAP_TOTAL_USD", "2")
    monkeypatch.setenv("GG_SPEND_CAP_RUN_USD", "0.10,openai=0.05")
    monkeypatch.setenv("GG_SPEND_GUARD_FAIL_MODE", "open")
    monkeypatch.setenv("GG_RUN_ID", "ci-nightly-1")
    s = SpendGuardSettings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.cap_daily_usd == {"openai": Decimal("0.50"), "gemini": Decimal("0.25")}
    assert s.cap_micros("day", "openai") == 500_000
    assert s.cap_micros("total", "anthropic") == 2_000_000
    assert s.cap_micros("run", "openai") == 50_000
    assert s.cap_micros("run", "gemini") == 100_000
    assert s.guard_fail_mode == "open"
    assert s.run_id == "ci-nightly-1"


def test_defaults_fail_closed_and_unknown_provider_gets_zero_cap() -> None:
    s = settings()
    assert s.guard_fail_mode == "closed"
    assert s.run_id is None
    assert s.cap_micros("day", "openai") == 1_000_000
    assert s.cap_micros("total", "gemini") == 3_000_000
    assert s.cap_micros("day", "brand-new") == 0


@pytest.mark.parametrize("raw", ["openai=abc", "=1.0", "openai=-1"])
def test_invalid_caps_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        settings(cap_daily_usd=raw)


async def test_in_memory_daily_cap_then_rollover(clock: FakeClock) -> None:
    telemetry = Telemetry()
    guard: ProviderSpendGuard = InMemorySpendGuard(
        settings(cap_daily_usd="openai=0.001", cap_total_usd="openai=0.002"), clock=clock, telemetry=telemetry
    )
    assert await guard.check("openai") is None
    await guard.record("openai", 600)
    assert await guard.check("openai") is None
    await guard.record("openai", 600)
    denial = await guard.check("openai")
    assert denial == SpendDenial("openai", "day", spent=1200, cap=1000, checked_at=clock.now())
    assert telemetry.denied == [("openai", "day")]

    clock.advance(3600)
    assert await guard.check("openai") is None
    await guard.record("openai", 900)
    denial = await guard.check("openai")
    assert denial is not None
    assert (denial.reason, denial.spent) == ("total", 2100)


async def test_unknown_provider_denied_and_zero_record_is_noop(clock: FakeClock) -> None:
    guard = InMemorySpendGuard(settings(), clock=clock)
    denial = await guard.check("together")
    assert denial is not None
    assert denial.reason == "day"
    await guard.record("openai", 0)
    assert guard.counters == {}
    with pytest.raises(ValueError, match="must not be negative"):
        await guard.record("openai", -1)


async def test_run_cap_only_applies_with_run_id(clock: FakeClock) -> None:
    without = InMemorySpendGuard(settings(cap_run_usd="0.000001"), clock=clock)
    await without.record("openai", 5)
    assert await without.check("openai") is None

    with_run = InMemorySpendGuard(settings(cap_run_usd="0.000001", run_id="r1"), clock=clock)
    await with_run.record("openai", 5)
    denial = await with_run.check("openai")
    assert denial is not None
    assert denial.reason == "run"
    assert "gg:spend:{p:openai}:run:r1" in with_run.counters


def test_denial_to_error() -> None:
    at = datetime(2026, 10, 5, 23, 0, tzinfo=UTC)
    error = SpendDenial("openai", "day", spent=1, cap=1, checked_at=at).to_error()
    assert isinstance(error, SpendCapReachedError)
    assert error.status == 503
    assert error.code == "provider_spend_cap"
    assert error.retry_after_s == 3600
    assert error.response_headers()["x-should-retry"] == "false"
    unavailable = SpendDenial("openai", "unavailable").to_error()
    assert "unavailable" in unavailable.message
    assert unavailable.retry_after_s is None


async def test_redis_counts_micros_without_ttl(clock: FakeClock) -> None:
    redis = fakeredis.FakeAsyncRedis()
    guard = RedisSpendGuard(redis, settings(cap_daily_usd="openai=0.000010", run_id="r1"), clock=clock)
    await guard.record("openai", 7)
    await guard.record("openai", 4)
    keys = ["gg:spend:{p:openai}:d:20261005", "gg:spend:{p:openai}:total", "gg:spend:{p:openai}:run:r1"]
    assert await redis.mget(keys) == [b"11", b"11", b"11"]
    for key in keys:
        assert await redis.ttl(key) == -1
    denial = await guard.check("openai")
    assert denial is not None
    assert (denial.reason, denial.spent, denial.cap) == ("day", 11, 10)

    clock.advance(DAY)
    assert await guard.check("openai") is None
    await redis.aclose()


async def test_redis_shares_state_across_guards(clock: FakeClock) -> None:
    server = fakeredis.FakeServer()
    a = RedisSpendGuard(
        fakeredis.FakeAsyncRedis(server=server), settings(cap_daily_usd="0.00001"), clock=clock
    )
    b = RedisSpendGuard(
        fakeredis.FakeAsyncRedis(server=server), settings(cap_daily_usd="0.00001"), clock=clock
    )
    await a.record("gemini", 10)
    denial = await b.check("gemini")
    assert denial is not None
    assert denial.spent == 10


async def test_redis_down_fails_closed(clock: FakeClock) -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    telemetry = Telemetry()
    guard = RedisSpendGuard(
        fakeredis.FakeAsyncRedis(server=server), settings(), clock=clock, telemetry=telemetry
    )
    denial = await guard.check("openai")
    assert denial is not None
    assert denial.reason == "unavailable"
    assert telemetry.denied == [("openai", "unavailable")]
    await guard.record("openai", 5)  # logged, never raises into the request


async def test_redis_down_fail_open_allows(clock: FakeClock) -> None:
    server = fakeredis.FakeServer()
    server.connected = False
    guard = RedisSpendGuard(
        fakeredis.FakeAsyncRedis(server=server), settings(guard_fail_mode="open"), clock=clock
    )
    assert await guard.check("openai") is None
