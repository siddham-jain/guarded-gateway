from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from gg.core.deployment import Deployment, PriceSchedule, Pricing
from gg.core.usage import UsageRecord
from gg.limits.cost import CostBreakdown, CostCalculator, cost_of

LUNA = Pricing(
    effective_from=date(2026, 1, 1),
    input=Decimal("0.10"),
    output=Decimal("0.50"),
    cached_input=Decimal("0.01"),
    cache_write=Decimal("0.125"),
)


def usage(
    *,
    input_tokens: int = 500,
    output_tokens: int = 200,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> UsageRecord:
    return UsageRecord(
        provider="openai",
        deployment_id="openai/gpt-6-luna",
        upstream_model="gpt-6-luna",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=reasoning_tokens,
    )


class Telemetry:
    def __init__(self) -> None:
        self.missing: list[tuple[str, str]] = []

    def pricing_missing(self, provider: str, deployment: str, /) -> None:
        self.missing.append((provider, deployment))

    def spend_denied(self, provider: str, reason: str, /) -> None:
        raise AssertionError("not expected")


def test_plain_usage_golden() -> None:
    # c9 lt-budget golden: 500 x 0.10 + 200 x 0.50 = 150 micro-usd
    cost = cost_of(usage(), LUNA)
    assert cost == CostBreakdown(
        input=50, cached_input=0, cache_write=0, output=100, effective_from=date(2026, 1, 1)
    )
    assert cost.total == 150
    assert cost.parts == {"input": 50, "cached_input": 0, "cache_write": 0, "output": 100}


def test_cached_and_cache_write_tokens_are_carved_out_of_input() -> None:
    cost = cost_of(usage(input_tokens=1000, cached_input_tokens=400, cache_write_tokens=100), LUNA)
    assert cost.input == 50  # 500 uncached x 0.10
    assert cost.cached_input == 4  # 400 x 0.01
    assert cost.cache_write == 13  # 100 x 0.125 = 12.5, ceil
    assert cost.total == 50 + 4 + 13 + 100


def test_missing_cache_prices_fall_back_to_input_price() -> None:
    pricing = Pricing(effective_from=date(2026, 1, 1), input=Decimal("2"), output=Decimal("10"))
    cost = cost_of(usage(input_tokens=1000, cached_input_tokens=300, cache_write_tokens=200), pricing)
    assert (cost.input, cost.cached_input, cost.cache_write) == (1000, 600, 400)


def test_reasoning_tokens_are_not_double_counted() -> None:
    with_reasoning = cost_of(usage(output_tokens=300, reasoning_tokens=250), LUNA)
    without = cost_of(usage(output_tokens=300), LUNA)
    assert with_reasoning == without
    assert with_reasoning.output == 150


@pytest.mark.parametrize(
    ("tokens", "price", "expected"),
    [(1, "0.10", 1), (10, "0.10", 1), (7, "0.15", 2), (3, "0.3333333", 1), (0, "5", 0)],
)
def test_ceil_rounding_per_component(tokens: int, price: str, expected: int) -> None:
    pricing = Pricing(effective_from=date(2026, 1, 1), input=Decimal(price), output=Decimal(0))
    assert cost_of(usage(input_tokens=tokens, output_tokens=0), pricing).input == expected


def test_over_reported_cache_tokens_clamp_uncached_to_zero() -> None:
    assert cost_of(usage(input_tokens=100, cached_input_tokens=150), LUNA).input == 0


def test_negative_tokens_rejected() -> None:
    with pytest.raises(ValueError, match="negative token count"):
        cost_of(usage(output_tokens=-1), LUNA)


def test_calculator_picks_the_period_effective_at_the_timestamp() -> None:
    later = Pricing(effective_from=date(2027, 1, 1), input=Decimal("0.20"), output=Decimal("1.00"))
    dep = Deployment(
        id="openai/gpt-6-luna",
        provider="openai",
        upstream_model="gpt-6-luna",
        pricing=PriceSchedule(periods=(later, LUNA), billed=False),
    )
    calc = CostCalculator()
    before = calc.cost(usage(), dep, datetime(2026, 12, 31, 23, 59, tzinfo=UTC))
    after = calc.cost(usage(), dep, datetime(2027, 1, 1, 0, 0, tzinfo=UTC))
    assert before is not None
    assert after is not None
    assert before.total == 150
    assert after.total == 300
    assert before.billed is False


def test_calculator_tolerates_missing_price() -> None:
    telemetry = Telemetry()
    dep = Deployment(id="mock/m", provider="mock", upstream_model="m")
    assert CostCalculator(telemetry).cost(usage(), dep, datetime(2026, 10, 5, tzinfo=UTC)) is None
    assert telemetry.missing == [("mock", "mock/m")]


def test_estimate_max_takes_the_dearest_deployment() -> None:
    def dep(dep_id: str, pricing: Pricing | None, *, billed: bool = True) -> Deployment:
        periods = (pricing,) if pricing else ()
        return Deployment(
            id=dep_id, provider="p", upstream_model=dep_id, pricing=PriceSchedule(periods, billed=billed)
        )

    at = datetime(2026, 10, 5, tzinfo=UTC)
    sol = Pricing(effective_from=date(2026, 1, 1), input=Decimal("2.00"), output=Decimal("10.00"))
    telemetry = Telemetry()
    calc = CostCalculator(telemetry)
    # luna: 1000 x 0.125 (cache write beats input) + 500 x 0.50 = 375; sol: 1000 x 2 + 500 x 10 = 7000
    assert calc.estimate_max(1_000, 500, [dep("luna", LUNA)], at) == 375
    deps = [dep("luna", LUNA), dep("sol", sol), dep("free", sol, billed=False), dep("unpriced", None)]
    assert calc.estimate_max(1_000, 500, deps, at) == 7_000
    assert telemetry.missing == [("p", "unpriced")]
    assert calc.estimate_max(1_000, 500, [], at) == 0
    with pytest.raises(ValueError, match="negative"):
        calc.estimate_max(-1, 0, deps, at)
