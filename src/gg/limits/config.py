from decimal import ROUND_FLOOR, Decimal
from typing import Annotated, Literal

from pydantic import PositiveFloat, PositiveInt, StringConstraints

from gg.config.loader import ConfigFile
from gg.core.keypolicy import BudgetPolicy, RateLimitPolicy
from gg.core.schema import StrictModel
from gg.limits.base import MICROS_PER_USD, BucketLimits, BudgetCaps, Micros


class RateLimitConfig(StrictModel):
    rpm_burst_window_s: PositiveFloat = 10
    tpm_burst_window_s: PositiveFloat = 60
    default_output_estimate: PositiveInt = 1024
    lease_grace_s: PositiveFloat = 30
    redis_timeout_ms: PositiveInt = 50
    breaker_failures: PositiveInt = 3
    breaker_cooldown_s: PositiveFloat = 5
    # open: a redis outage falls back to a local in-process bucket; closed: 503 ratelimit_unavailable
    fail_mode: Literal["open", "closed"] = "open"

    def bucket_limits(self, policy: RateLimitPolicy) -> BucketLimits:
        rpm = policy.rpm or 0
        tpm = policy.tpm or 0
        return BucketLimits(
            rpm=rpm,
            rpm_capacity=max(1.0, rpm * self.rpm_burst_window_s / 60) if rpm else 0.0,
            tpm=tpm,
            tpm_capacity=max(1.0, tpm * self.tpm_burst_window_s / 60) if tpm else 0.0,
            max_concurrent=policy.max_concurrent or 0,
        )

    @property
    def bucket_ttl_ms(self) -> int:
        # long enough for a bucket in full debt (-capacity) to refill before the key expires
        return int(max(120.0, 2 * max(self.rpm_burst_window_s, self.tpm_burst_window_s)) * 1000)


class BudgetConfig(StrictModel):
    redis_timeout_ms: PositiveInt = 200
    hold_grace_s: PositiveFloat = 60
    sweep_limit: PositiveInt = 20
    tombstone_s: PositiveInt = 86_400


class LimitsConfig(StrictModel):
    version: Literal[1] = 1
    redis_prefix: Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]{1,32}$")] = "gg"
    rate_limits: RateLimitConfig = RateLimitConfig()
    budgets: BudgetConfig = BudgetConfig()


LIMITS_CONFIG_FILE = ConfigFile("limits", "limits.yaml", LimitsConfig)


def usd_to_micros(usd: Decimal) -> Micros:
    return int((usd * MICROS_PER_USD).to_integral_value(rounding=ROUND_FLOOR))


def budget_caps(policy: BudgetPolicy) -> BudgetCaps:
    return BudgetCaps(
        daily=usd_to_micros(policy.daily_usd) if policy.daily_usd is not None else None,
        monthly=usd_to_micros(policy.monthly_usd) if policy.monthly_usd is not None else None,
    )
