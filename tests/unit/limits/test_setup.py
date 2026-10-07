import hashlib
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from gg.config.loader import load_file
from gg.core.context import RequestContext
from gg.core.keypolicy import BudgetPolicy
from gg.limits.base import BudgetCaps
from gg.limits.bucket import LocalRateLimiter
from gg.limits.config import LIMITS_CONFIG_FILE, LimitsConfig, budget_caps, usd_to_micros
from gg.limits.cost import CostCalculator
from gg.limits.ledger import InMemoryBudgetLedger, RedisBudgetLedger
from gg.limits.lua import ADJUST, ADMIT, FINISH, RESERVE, SETTLE
from gg.limits.redis_limiter import ResilientRateLimiter
from gg.limits.setup import build_limits
from gg.limits.stage import LimitsStage
from gg.pipeline.stage import PipelineResult
from gg.providers.usage import estimate_prompt_tokens
from tests.unit.limits.support import LUNA, FakeCatalog, down_redis, fake_clock, fake_redis, make_ctx

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def test_repo_config_matches_the_defaults() -> None:
    loaded = load_file(CONFIG_DIR / LIMITS_CONFIG_FILE.path, LIMITS_CONFIG_FILE.model)
    assert loaded == LimitsConfig()
    assert loaded.rate_limits.bucket_ttl_ms == 120_000


def test_config_rejects_typos_and_bad_prefixes() -> None:
    with pytest.raises(ValidationError):
        LimitsConfig.model_validate({"rate_limits": {"rpm_burst_windows": 10}})
    with pytest.raises(ValidationError):
        LimitsConfig.model_validate({"redis_prefix": "GG:bad"})


def test_budget_caps_convert_usd_to_floor_micros() -> None:
    assert usd_to_micros(Decimal("0.0000019")) == 1
    policy = BudgetPolicy(daily_usd=Decimal("1.00"), monthly_usd=Decimal("5.5"))
    assert budget_caps(policy) == BudgetCaps(daily=1_000_000, monthly=5_500_000)
    assert not budget_caps(BudgetPolicy()).enabled


async def test_without_redis_everything_is_in_process() -> None:
    built = build_limits(
        LimitsConfig(),
        redis=None,
        clock=fake_clock(),
        catalog=FakeCatalog({}),
        costs=CostCalculator(),
        estimate_prompt=estimate_prompt_tokens,
    )
    assert isinstance(built.stage, LimitsStage)
    assert built.stage.name == "limits"
    assert isinstance(built.limiter, LocalRateLimiter)
    assert isinstance(built.ledger, InMemoryBudgetLedger)
    await built.start()
    await built.stop()


async def test_with_redis_start_loads_the_scripts_and_the_stage_works_end_to_end() -> None:
    redis = fake_redis()
    clock = fake_clock()
    built = build_limits(
        LimitsConfig(),
        redis=redis,
        clock=clock,
        catalog=FakeCatalog({"mock/luna": [LUNA]}),
        costs=CostCalculator(),
        estimate_prompt=estimate_prompt_tokens,
    )
    assert isinstance(built.limiter, ResilientRateLimiter)
    assert isinstance(built.ledger, RedisBudgetLedger)
    await built.start()
    shas = [
        hashlib.sha1(script.encode(), usedforsecurity=False).hexdigest()
        for script in (ADMIT, FINISH, RESERVE, ADJUST, SETTLE)
    ]
    assert await redis.script_exists(*shas) == [True] * 5

    ctx = make_ctx(clock, key={"budget": {"daily_usd": "1.00"}}, model="mock/luna", max_completion_tokens=100)

    async def cache_hit(c: RequestContext) -> PipelineResult:
        c.cache_status = "exact_hit"
        return PipelineResult(source="exact_cache", response=None)

    await built.stage(ctx, cache_hit)
    assert ctx.response_headers["x-ratelimit-remaining-requests"] == "9"
    assert ctx.response_headers["x-gg-budget-period"] == "day"
    hold_key = "gg:bud:{k:test-key}:d:20261005"
    assert int(await redis.hget(hold_key, "reserved") or 0) > 0
    await ctx.finalizers.run(timeout_s=1)
    raw = await redis.hgetall(hold_key)
    assert {k.decode(): int(v) for k, v in raw.items()} == {"spent": 0, "reserved": 0}


async def test_start_survives_a_redis_outage() -> None:
    built = build_limits(
        LimitsConfig(),
        redis=down_redis(),
        clock=fake_clock(),
        catalog=FakeCatalog({}),
        costs=CostCalculator(),
        estimate_prompt=estimate_prompt_tokens,
    )
    await built.start()
