import random
from datetime import timedelta

import pytest

from gg.core.clock import FakeClock
from gg.core.errors import ProviderError
from gg.reliability.breaker import Cooldown
from gg.reliability.policy import (
    Decision,
    FullJitter,
    RetryBudget,
    RetryConfig,
    RetryPolicy,
    counts_against_breaker,
)
from tests.unit.reliability.fakes import err


class ZeroBackoff:
    def delay(self, retry_number: int, /) -> float:
        return 0.1


def policy(clock: FakeClock, **cfg: object) -> RetryPolicy:
    config = RetryConfig(**cfg)  # type: ignore[arg-type]
    return RetryPolicy(clock, config, backoff=ZeroBackoff(), rng=random.Random(1))  # noqa: S311


@pytest.mark.parametrize(
    ("error", "retries", "expected"),
    [
        (err("retryable", status=500), 0, Decision("retry", 0.1)),
        (err("retryable", status=500), 1, Decision("fallback")),
        (err("retryable", status=0, code="connect_error"), 1, Decision("retry", 0.1)),
        (err("retryable", status=0, code="connect_error"), 2, Decision("fallback")),
        (err("retryable", status=504, code="ttft_timeout"), 0, Decision("fallback")),
        (err("fallback", status=529, code="overloaded"), 0, Decision("fallback")),
        (err("fallback", status=0, code="capability_mismatch"), 0, Decision("fallback")),
        (err("fallback", status=400, code="context_length"), 0, Decision("fallback_larger_context")),
        (
            err("fallback", status=404, code="model_not_found"),
            0,
            Decision("fallback", cooldown=Cooldown(600, "model_not_found")),
        ),
        (err("client", status=400), 0, Decision("fail")),
        (err("content_filter", status=400), 0, Decision("fail")),
        (err("auth", status=401), 0, Decision("fallback", cooldown=Cooldown(600, "auth"))),
        (
            err("auth", status=429, code="billing", scope="provider"),
            0,
            Decision("fallback", cooldown=Cooldown(3600, "billing")),
        ),
        (err("quota_minute", status=429), 0, Decision("fallback")),
        (
            err("quota_minute", status=429, retry_after_s=30),
            0,
            Decision("fallback", cooldown=Cooldown(30, "rate_limited")),
        ),
        (
            err("quota_minute", status=429, retry_after_s=600),
            0,
            Decision("fallback", cooldown=Cooldown(60, "rate_limited")),
        ),
    ],
)
def test_decision_table(clock: FakeClock, error: ProviderError, retries: int, expected: Decision) -> None:
    assert policy(clock).decide(error, "a/m", retries, remaining_s=30) == expected


def test_retry_after_within_wait_sleeps_with_jitter(clock: FakeClock) -> None:
    decision = policy(clock).decide(err("quota_minute", status=429, retry_after_s=2), "a/m", 0, 30)
    assert decision.action == "retry"
    assert 2.0 <= decision.delay_s <= 2.25


def test_retry_after_past_deadline_falls_back_without_cooldown(clock: FakeClock) -> None:
    decision = policy(clock).decide(err("quota_minute", status=429, retry_after_s=3), "a/m", 0, 4)
    assert decision == Decision("fallback")


def test_backoff_past_deadline_falls_back(clock: FakeClock) -> None:
    assert policy(clock).decide(err("retryable", status=500), "a/m", 0, 1.55) == Decision("fallback")


def test_quota_day_cools_down_until_reset(clock: FakeClock) -> None:
    reset = clock.now() + timedelta(hours=5)
    decision = policy(clock).decide(err("quota_day", status=429, quota_reset_at=reset), "a/m", 0, 30)
    assert decision.action == "fallback"
    assert decision.cooldown is not None
    assert decision.cooldown.seconds == pytest.approx(5 * 3600)


def test_content_filter_fallback_is_configurable(clock: FakeClock) -> None:
    p = policy(clock, fallback_on_content_filter=True)
    assert p.decide(err("content_filter", status=400), "a/m", 0, 30) == Decision("fallback")


def test_exhausted_budget_turns_retry_into_fallback(clock: FakeClock) -> None:
    p = policy(clock)
    error = err("retryable", status=500)
    for _ in range(3):
        p.on_attempt("a/m")
        assert p.decide(error, "a/m", 0, 30).action == "retry"
    assert p.decide(error, "a/m", 0, 30) == Decision("fallback")
    assert p.decide(error, "b/m", 0, 30).action == "retry"


@pytest.mark.parametrize(
    ("error", "counts"),
    [
        (err("retryable", status=500), True),
        (err("retryable", status=504, code="ttft_timeout"), True),
        (err("retryable", status=0, code="pool_timeout"), False),
        (err("retryable", status=409, code="conflict"), False),
        (err("fallback", status=529, code="overloaded"), True),
        (err("fallback", status=0, code="capability_mismatch"), False),
        (err("fallback", status=400, code="context_length"), False),
        (err("client", status=400), False),
        (err("quota_minute", status=429), False),
        (err("auth", status=401), False),
    ],
)
def test_breaker_accounting(error: ProviderError, counts: bool) -> None:
    assert counts_against_breaker(error) is counts


def test_full_jitter_bounds() -> None:
    jitter = FullJitter(0.25, 4.0, random.Random(42))  # noqa: S311
    for n, bound in [(0, 0.25), (1, 0.5), (3, 2.0), (6, 4.0)]:
        samples = [jitter.delay(n) for _ in range(10_000)]
        assert all(0 <= s <= bound for s in samples)
        assert sum(samples) / len(samples) == pytest.approx(bound / 2, rel=0.05)


def test_retry_budget_floor_and_ratio(clock: FakeClock) -> None:
    budget = RetryBudget(ratio=0.1, min_per_window=3, window_s=60, clock=clock)
    assert [budget.try_spend() for _ in range(4)] == [True, True, True, False]

    for _ in range(100):
        budget.on_request()
    assert [budget.try_spend() for _ in range(8)] == [True] * 7 + [False]


def test_retry_budget_window_rolls_over(clock: FakeClock) -> None:
    budget = RetryBudget(ratio=0.1, min_per_window=3, window_s=60, clock=clock)
    for _ in range(3):
        assert budget.try_spend()
    assert not budget.try_spend()
    clock.advance(30)
    assert not budget.try_spend()
    clock.advance(31)
    assert budget.try_spend()
