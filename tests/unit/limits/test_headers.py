from datetime import UTC, datetime

import pytest

from gg.core.keypolicy import RateLimitPolicy
from gg.limits.base import BudgetStatus, LimitResult
from gg.limits.config import RateLimitConfig
from gg.limits.headers import budget_headers, cost_headers, go_duration, rate_limit_headers


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (-3, "0s"),
        (0.0004, "0s"),
        (0.85, "850ms"),
        (1, "1s"),
        (1.5, "1.5s"),
        (59.999, "59.999s"),
        (360, "6m0s"),
        (61.25, "1m1.25s"),
        (3_600, "1h0m0s"),
        (3_725, "1h2m5s"),
    ],
)
def test_go_durations(seconds: float, expected: str) -> None:
    assert go_duration(seconds) == expected


def test_rate_limit_headers_only_for_enabled_limits_and_never_negative() -> None:
    limits = RateLimitConfig().bucket_limits(RateLimitPolicy(rpm=60, tpm=None, max_concurrent=3))
    result = LimitResult(
        allowed=False, reason="rpm", limits=limits, remaining_requests=0, reset_requests_s=0.85, degraded=True
    )
    assert rate_limit_headers(result) == {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "850ms",
        "x-gg-ratelimit-degraded": "local",
    }


def test_budget_headers_with_soft_warning() -> None:
    reset = datetime(2026, 10, 6, tzinfo=UTC)
    calm = BudgetStatus("day", 500_000, 418_766, reset)
    assert budget_headers(calm, soft_limit_pct=0.9) == {
        "x-gg-budget-limit-usd": "0.500000",
        "x-gg-budget-remaining-usd": "0.081234",
        "x-gg-budget-period": "day",
    }
    assert budget_headers(calm, soft_limit_pct=0.8)["x-gg-budget-warning"] == "soft_limit"
    over = BudgetStatus("month", 1_000, 1_200, reset)
    assert budget_headers(over, soft_limit_pct=0.8)["x-gg-budget-remaining-usd"] == "0.000000"


def test_cost_headers() -> None:
    assert cost_headers(412, "reported") == {"x-gg-cost-usd": "0.000412", "x-gg-usage-source": "reported"}
    assert cost_headers(0, None) == {"x-gg-cost-usd": "0.000000"}
