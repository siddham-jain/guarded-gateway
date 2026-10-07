from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from gg.core.cache_types import CacheStatus
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.deployment import Deployment
from gg.core.errors import QuotaExceededError, RateLimitedError
from gg.core.schema import AssistantMessage, ChatResponse, Choice
from gg.core.usage import TokenEstimates, UsageRecord
from gg.limits.base import (
    LIMITS_STATE,
    BudgetLedger,
    BudgetUnavailableError,
    LimitsState,
    TokensExceedLimitError,
)
from gg.limits.bucket import LocalRateLimiter
from gg.limits.config import LimitsConfig
from gg.limits.cost import CostCalculator
from gg.limits.ledger import InMemoryBudgetLedger, RedisBudgetLedger
from gg.limits.stage import LimitsStage
from gg.pipeline.stage import PipelineResult
from tests.unit.limits.support import (
    HAIKU,
    LUNA,
    SOL,
    FakeCatalog,
    RecordingHooks,
    down_redis,
    fake_clock,
    make_ctx,
)

PROMPT_TOKENS = 500
DAY = "gg:bud:{k:test-key}:d:20261005"
CATALOG = FakeCatalog({"mock/luna": [LUNA], "gg/auto": [LUNA, SOL], "gg/mixed": [LUNA, HAIKU]})
NO_BUDGET: dict[str, Any] = {"budget": {}}
# keys.yaml defaults: $1.00 a day, $5.00 a month
DEFAULT_KEY: dict[str, Any] = {"budget": {"daily_usd": "1.00", "monthly_usd": "5.00"}}


@dataclass
class Rig:
    clock: FakeClock = field(default_factory=fake_clock)
    hooks: RecordingHooks = field(default_factory=RecordingHooks)
    config: LimitsConfig = field(default_factory=LimitsConfig)
    ledger_override: BudgetLedger | None = None

    def __post_init__(self) -> None:
        self.limiter = LocalRateLimiter(self.clock)
        self.memory = InMemoryBudgetLedger(self.config.budgets, clock=self.clock)
        self.stage = LimitsStage(
            self.config,
            limiter=self.limiter,
            ledger=self.ledger_override or self.memory,
            catalog=CATALOG,
            costs=CostCalculator(),
            estimate_prompt=lambda _: PROMPT_TOKENS,
            hooks=self.hooks,
        )

    def ctx(self, *, key: dict[str, Any] | None = None, **request: Any) -> RequestContext:
        request.setdefault("model", "mock/luna")
        return make_ctx(self.clock, key={**DEFAULT_KEY, **(key or {})}, **request)

    async def run(
        self,
        ctx: RequestContext,
        *,
        served: Deployment | None = LUNA,
        usage: tuple[int, int] | None = (500, 200),
        cache: CacheStatus = "miss",
    ) -> PipelineResult:
        async def terminal(c: RequestContext) -> PipelineResult:
            c.cache_status = cache
            c.served_by = served
            if served is not None and usage is not None:
                c.usage = UsageRecord(
                    provider=served.provider,
                    deployment_id=served.id,
                    upstream_model=served.upstream_model,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                )
            return PipelineResult(source="upstream", response=RESPONSE)

        return await self.stage(ctx, terminal)


RESPONSE = ChatResponse(
    id="chatcmpl-1",
    created=1,
    model="m",
    choices=(Choice(index=0, message=AssistantMessage(content="hi"), finish_reason="stop"),),
)


def state(ctx: RequestContext) -> LimitsState:
    found = ctx.get(LIMITS_STATE)
    assert found is not None
    return found


def finalizer_names(ctx: RequestContext) -> Sequence[str]:
    return [name for _, _, name, _ in ctx.finalizers._items]


async def test_success_sets_estimates_headers_and_finalizers() -> None:
    rig = Rig()
    ctx = rig.ctx(max_completion_tokens=300)
    await rig.run(ctx)
    assert ctx.estimates == TokenEstimates(prompt_tokens=500, max_completion_tokens=300)
    headers = ctx.response_headers
    assert headers["x-ratelimit-limit-requests"] == "60"
    assert headers["x-ratelimit-remaining-requests"] == "9"
    assert headers["x-ratelimit-reset-requests"] == "1s"
    assert headers["x-ratelimit-limit-tokens"] == "200000"
    assert headers["x-ratelimit-remaining-tokens"] == "199200"
    assert headers["x-ratelimit-reset-tokens"] == "240ms"
    # hold = 500 x 0.10 + 300 x 0.50 = 200 micro-usd against the $1.00 daily default
    assert headers["x-gg-budget-limit-usd"] == "1.000000"
    assert headers["x-gg-budget-remaining-usd"] == "0.999800"
    assert headers["x-gg-budget-period"] == "day"
    assert "x-gg-budget-warning" not in headers
    # non-stream responses carry the served attempt's cost: 500 x 0.10 + 200 x 0.50 = 150 micro-usd
    assert headers["x-gg-cost-usd"] == "0.000150"
    assert headers["x-gg-usage-source"] == "reported"
    assert finalizer_names(ctx) == ["limits", "budget"]
    assert "x-gg-ratelimit-degraded" not in headers


async def test_finalizers_settle_actual_cost_and_reconcile_tokens() -> None:
    rig = Rig()
    ctx = rig.ctx(max_completion_tokens=300)
    await rig.run(ctx)
    hold = state(ctx).hold
    assert hold is not None
    assert hold.amount == 200
    assert rig.memory.reserved(DAY) == 200
    await ctx.finalizers.run(timeout_s=1)
    assert (rig.memory.spent(DAY), rig.memory.reserved(DAY)) == (150, 0)
    assert hold.state == "settled"
    # 800 estimated, 700 used: the next request sees 100 refunded
    nxt = rig.ctx(max_completion_tokens=300)
    await rig.run(nxt)
    assert nxt.response_headers["x-ratelimit-remaining-tokens"] == str(200_000 - 700 - 800)


async def test_cache_hit_settles_at_zero_and_refunds_tokens() -> None:
    rig = Rig()
    ctx = rig.ctx(max_completion_tokens=300)
    await rig.run(ctx, cache="exact_hit")
    assert ctx.response_headers["x-gg-cost-usd"] == "0.000000"
    assert "x-gg-usage-source" not in ctx.response_headers
    await ctx.finalizers.run(timeout_s=1)
    assert (rig.memory.spent(DAY), rig.memory.reserved(DAY)) == (0, 0)
    nxt = rig.ctx(max_completion_tokens=300)
    await rig.run(nxt, cache="semantic_hit")
    assert nxt.response_headers["x-ratelimit-remaining-tokens"] == str(200_000 - 800)


async def test_unserved_request_pays_nothing_and_cut_stream_pays_the_hold() -> None:
    rig = Rig()
    failed = rig.ctx(max_completion_tokens=300)
    await rig.run(failed, served=None)
    await failed.finalizers.run(timeout_s=1)
    assert rig.memory.spent(DAY) == 0
    cut = rig.ctx(max_completion_tokens=300)
    await rig.run(cut, usage=None)
    assert "x-gg-cost-usd" not in cut.response_headers
    await cut.finalizers.run(timeout_s=1)
    assert rig.memory.spent(DAY) == 200


async def test_rpm_rejection_is_429_with_retry_after() -> None:
    rig = Rig()
    key = {"rate_limits": {"rpm": 6, "tpm": None, "max_concurrent": None}, **NO_BUDGET}
    await rig.run(rig.ctx(key=key))
    ctx = rig.ctx(key=key)
    with pytest.raises(RateLimitedError) as info:
        await rig.run(ctx)
    err = info.value
    assert (err.status, err.code, err.type) == (429, "rate_limit_exceeded", "rate_limit_error")
    assert err.message == (
        "Rate limit reached for key 'test-key' on requests per min (RPM): limit 6. Please try again in 10.0s."
    )
    assert err.response_headers()["retry-after"] == "10"
    assert err.response_headers()["x-should-retry"] == "true"
    assert ctx.response_headers["x-ratelimit-remaining-requests"] == "0"
    assert ctx.response_headers["x-ratelimit-reset-requests"] == "10s"
    assert "x-ratelimit-limit-tokens" not in ctx.response_headers
    assert rig.hooks.rejections == ["rpm"]
    assert finalizer_names(ctx) == []


async def test_tpm_rejection_and_request_too_large() -> None:
    rig = Rig()
    key = {"rate_limits": {"rpm": None, "tpm": 1_000, "max_concurrent": None}, **NO_BUDGET}
    await rig.run(rig.ctx(key=key, max_completion_tokens=400))
    with pytest.raises(RateLimitedError) as info:
        await rig.run(rig.ctx(key=key, max_completion_tokens=400))
    assert info.value.code == "rate_limit_exceeded"
    assert "tokens per min (TPM): limit 1000, requested 900" in info.value.message
    with pytest.raises(TokensExceedLimitError) as too_big:
        await rig.run(rig.ctx(key=key, max_completion_tokens=501))
    assert (too_big.value.status, too_big.value.code) == (429, "tokens_exceed_limit")
    assert too_big.value.response_headers()["x-should-retry"] == "false"
    assert "retry-after" not in too_big.value.response_headers()
    assert rig.hooks.rejections == ["tpm", "tokens_exceed_limit"]


async def test_concurrency_limit_and_release_on_finish() -> None:
    rig = Rig()
    key = {"rate_limits": {"rpm": None, "tpm": None, "max_concurrent": 1}, **NO_BUDGET}
    first = rig.ctx(key=key)
    await rig.run(first)
    with pytest.raises(RateLimitedError) as info:
        await rig.run(rig.ctx(key=key))
    assert info.value.code == "concurrency_limit_exceeded"
    assert info.value.response_headers()["retry-after"] == "1"
    await first.finalizers.run(timeout_s=1)
    await rig.run(rig.ctx(key=key))


async def test_default_output_allowance_and_n() -> None:
    rig = Rig()
    ctx = rig.ctx(key={"limits": {"max_completion_tokens": 512, "max_n": 4}}, n=3)
    await rig.run(ctx)
    assert ctx.estimates == TokenEstimates(prompt_tokens=500, max_completion_tokens=1_536)
    unset = rig.ctx(key={"limits": {"max_completion_tokens": None}})
    await rig.run(unset)
    assert unset.estimates == TokenEstimates(prompt_tokens=500, max_completion_tokens=1_024)


async def test_worst_case_is_the_dearest_reachable_deployment() -> None:
    rig = Rig()
    ctx = rig.ctx(model="gg/auto", max_completion_tokens=1_000)
    await rig.run(ctx)
    # sol: 500 x 2.00 + 1000 x 10.00 = 11000 micro-usd (luna would be 550)
    hold = state(ctx).hold
    assert hold is not None
    assert hold.amount == 11_000


async def test_worst_case_skips_providers_the_key_cannot_use() -> None:
    rig = Rig()
    ctx = rig.ctx(model="gg/mixed", max_completion_tokens=1_000, key={"allowed_providers": ["anthropic"]})
    await rig.run(ctx)
    # haiku prices the prompt at its cache-write rate: 500 x 1.25 + 1000 x 5.00 = 5625
    hold = state(ctx).hold
    assert hold is not None
    assert hold.amount == 5_625


@pytest.mark.parametrize(
    ("key", "gg", "expected"),
    [
        ({"budget": {"daily_usd": "1", "max_request_usd": "0.004"}}, None, 4_000),
        ({}, {"max_cost_usd": "0.003"}, 3_000),
        ({"budget": {"daily_usd": "1", "max_request_usd": "0.004"}}, {"max_cost_usd": "0.005"}, 4_000),
    ],
)
async def test_hold_is_capped_by_per_request_limits(
    key: dict[str, Any], gg: dict[str, Any] | None, expected: int
) -> None:
    rig = Rig()
    ctx = rig.ctx(model="gg/auto", max_completion_tokens=1_000, key=key, gg=gg)
    await rig.run(ctx)
    hold = state(ctx).hold
    assert hold is not None
    assert hold.amount == expected


async def test_soft_warning_then_hard_402() -> None:
    rig = Rig()
    key = {"budget": {"daily_usd": "0.001", "soft_limit_pct": 0.5}}
    first = rig.ctx(key=key, max_completion_tokens=300)
    await rig.run(first)
    assert "x-gg-budget-warning" not in first.response_headers
    second = rig.ctx(key=key, max_completion_tokens=300)
    await rig.run(second)
    # 400 of 1000 reserved is below 0.5; the third hold lands at 0.6
    third = rig.ctx(key=key, max_completion_tokens=300)
    await rig.run(third)
    assert third.response_headers["x-gg-budget-warning"] == "soft_limit"
    assert rig.hooks.events == ["soft_limit"]
    for ctx in (first, second, third):
        await ctx.finalizers.run(timeout_s=1)
    assert rig.memory.spent(DAY) == 450
    blocked = rig.ctx(key=key, max_completion_tokens=1_100)
    with pytest.raises(QuotaExceededError) as info:
        await rig.run(blocked)
    err = info.value
    assert (err.status, err.code) == (402, "budget_exceeded")
    assert err.response_headers()["x-should-retry"] == "false"
    assert blocked.response_headers["x-gg-budget-remaining-usd"] == "0.000550"
    assert "x-gg-budget-warning" not in blocked.response_headers
    assert rig.hooks.rejections == ["budget"]
    assert rig.hooks.events == ["soft_limit", "hard_limit"]
    # the 402 still releases the rate-limit lease and refunds its tokens
    assert finalizer_names(blocked) == ["limits"]
    await blocked.finalizers.run(timeout_s=1)
    probe = rig.ctx(key=NO_BUDGET, max_completion_tokens=300)
    await rig.run(probe)
    assert probe.response_headers["x-ratelimit-remaining-tokens"] == str(200_000 - 3 * 700 - 800)


async def test_budget_outage_fails_closed_with_503() -> None:
    clock = fake_clock()
    rig = Rig(
        clock=clock, ledger_override=RedisBudgetLedger(down_redis(), LimitsConfig().budgets, clock=clock)
    )
    ctx = rig.ctx()
    with pytest.raises(BudgetUnavailableError) as info:
        await rig.run(ctx)
    assert (info.value.status, info.value.code) == (503, "budget_unavailable")
    assert rig.hooks.events == ["fail_closed"]
    assert rig.hooks.rejections == ["budget_unavailable"]


async def test_budget_outage_fails_open_when_the_key_opts_in() -> None:
    clock = fake_clock()
    rig = Rig(
        clock=clock, ledger_override=RedisBudgetLedger(down_redis(), LimitsConfig().budgets, clock=clock)
    )
    ctx = rig.ctx(key={"budget": {"daily_usd": "1", "fail_mode": "open"}})
    await rig.run(ctx)
    assert rig.hooks.events == ["fail_open"]
    assert "x-gg-budget-limit-usd" not in ctx.response_headers
    assert finalizer_names(ctx) == ["limits"]


async def test_keys_without_limits_skip_everything() -> None:
    rig = Rig()
    key = {"rate_limits": {"rpm": None, "tpm": None, "max_concurrent": None}, **NO_BUDGET}
    ctx = rig.ctx(key=key)
    await rig.run(ctx)
    assert finalizer_names(ctx) == []
    assert not any(h.startswith(("x-ratelimit", "x-gg-budget")) for h in ctx.response_headers)
    assert state(ctx).hold is None
