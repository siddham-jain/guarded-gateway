from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import ROUND_CEILING, Decimal

import structlog

from gg.core.deployment import Deployment, Pricing
from gg.core.usage import UsageRecord
from gg.limits.base import LimitsTelemetry, Micros, NullLimitsTelemetry

log = structlog.get_logger("gg.limits.cost")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """integer micro-usd per component; reasoning tokens are priced inside output"""

    input: Micros
    cached_input: Micros
    cache_write: Micros
    output: Micros
    effective_from: date
    billed: bool = True

    @property
    def total(self) -> Micros:
        return self.input + self.cached_input + self.cache_write + self.output

    @property
    def parts(self) -> Mapping[str, Micros]:
        return {
            "input": self.input,
            "cached_input": self.cached_input,
            "cache_write": self.cache_write,
            "output": self.output,
        }


def _micros(tokens: int, usd_per_million: Decimal) -> Micros:
    # usd per 1m tokens times tokens is exactly micro-usd; ceil so gg never under-counts
    return int((tokens * usd_per_million).to_integral_value(rounding=ROUND_CEILING))


def cost_of(usage: UsageRecord, pricing: Pricing) -> CostBreakdown:
    counts = (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.cache_write_tokens,
        usage.output_tokens,
        usage.reasoning_tokens,
    )
    if any(n < 0 for n in counts):
        raise ValueError(f"negative token count in usage for {usage.deployment_id}")
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens)
    cached_rate = pricing.input if pricing.cached_input is None else pricing.cached_input
    write_rate = pricing.input if pricing.cache_write is None else pricing.cache_write
    return CostBreakdown(
        input=_micros(uncached, pricing.input),
        cached_input=_micros(usage.cached_input_tokens, cached_rate),
        cache_write=_micros(usage.cache_write_tokens, write_rate),
        output=_micros(usage.output_tokens, pricing.output),
        effective_from=pricing.effective_from,
    )


class CostCalculator:
    """prices usage with the deployment's schedule at a timestamp; a missing price yields None"""

    def __init__(self, telemetry: LimitsTelemetry | None = None) -> None:
        self._telemetry = telemetry or NullLimitsTelemetry()

    def cost(self, usage: UsageRecord, deployment: Deployment, at: datetime, /) -> CostBreakdown | None:
        pricing = deployment.pricing.at(at)
        if pricing is None:
            self._telemetry.pricing_missing(deployment.provider, deployment.id)
            log.warning("cost.price_missing", deployment=deployment.id, at=at.isoformat())
            return None
        return replace(cost_of(usage, pricing), billed=deployment.pricing.billed)

    def estimate_max(
        self, prompt_tokens: int, max_output_tokens: int, deployments: Sequence[Deployment], at: datetime, /
    ) -> Micros:
        """worst case over the deployments; prompt priced at the dearer of input and cache write.

        key budgets charge list price even on unbilled deployments; unpriced ones are skipped (and reported)
        """
        if prompt_tokens < 0 or max_output_tokens < 0:
            raise ValueError("token estimates must not be negative")
        worst = 0
        for deployment in deployments:
            pricing = deployment.pricing.at(at)
            if pricing is None:
                self._telemetry.pricing_missing(deployment.provider, deployment.id)
                continue
            input_rate = max(pricing.input, pricing.cache_write or pricing.input)
            worst = max(
                worst, _micros(prompt_tokens, input_rate) + _micros(max_output_tokens, pricing.output)
            )
        return worst
