"""what the composition root needs: the LimitsStage plus script loading at startup"""

from dataclasses import dataclass

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gg.core.clock import Clock
from gg.limits.base import BudgetLedger, LimitsHooks, NullLimitsHooks, RateLimiter
from gg.limits.bucket import LocalRateLimiter
from gg.limits.config import LimitsConfig
from gg.limits.cost import CostCalculator
from gg.limits.ledger import InMemoryBudgetLedger, RedisBudgetLedger
from gg.limits.redis_limiter import RedisRateLimiter, ResilientRateLimiter
from gg.limits.stage import LimitsStage, PromptEstimator
from gg.providers.base import ModelCatalog

log = structlog.get_logger("gg.limits")


@dataclass(frozen=True, slots=True)
class BuiltLimits:
    stage: LimitsStage
    limiter: RateLimiter
    ledger: BudgetLedger
    redis_limiter: RedisRateLimiter | None = None
    redis_ledger: RedisBudgetLedger | None = None

    async def start(self) -> None:
        """preloads the lua scripts; a failure is only logged because EVALSHA reloads them on NOSCRIPT"""
        if self.redis_limiter is None or self.redis_ledger is None:
            return
        try:
            shas = [*await self.redis_limiter.load(), *await self.redis_ledger.load()]
        except (RedisError, OSError) as exc:
            log.warning("limits.script_load_failed", error=type(exc).__name__)
            return
        log.info("limits.scripts_loaded", shas=shas)

    async def stop(self) -> None:
        # the redis client belongs to the composition root, which closes it
        return None


def build_limits(
    config: LimitsConfig,
    *,
    redis: Redis | None,
    clock: Clock,
    catalog: ModelCatalog,
    costs: CostCalculator,
    estimate_prompt: PromptEstimator,
    metrics_hooks: LimitsHooks | None = None,
) -> BuiltLimits:
    hooks = metrics_hooks or NullLimitsHooks()
    local = LocalRateLimiter(clock)
    redis_limiter: RedisRateLimiter | None = None
    redis_ledger: RedisBudgetLedger | None = None
    limiter: RateLimiter
    ledger: BudgetLedger
    if redis is None:
        log.warning(
            "limits.in_memory", reason="no redis; limits and budgets are per process and reset on restart"
        )
        limiter = local
        ledger = InMemoryBudgetLedger(config.budgets, clock=clock, prefix=config.redis_prefix, hooks=hooks)
    else:
        redis_limiter = RedisRateLimiter(redis, config.rate_limits, prefix=config.redis_prefix)
        limiter = ResilientRateLimiter(redis_limiter, local, config.rate_limits, clock=clock, hooks=hooks)
        redis_ledger = RedisBudgetLedger(
            redis, config.budgets, clock=clock, prefix=config.redis_prefix, hooks=hooks
        )
        ledger = redis_ledger
    stage = LimitsStage(
        config,
        limiter=limiter,
        ledger=ledger,
        catalog=catalog,
        costs=costs,
        estimate_prompt=estimate_prompt,
        hooks=hooks,
    )
    return BuiltLimits(stage, limiter, ledger, redis_limiter, redis_ledger)
