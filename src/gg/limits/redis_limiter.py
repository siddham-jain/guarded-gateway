import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gg.core.clock import Clock
from gg.limits.base import (
    BucketLimits,
    Lease,
    LimitReason,
    LimitResult,
    LimitsHooks,
    NullLimitsHooks,
    RateLimiter,
    RateLimiterUnavailableError,
)
from gg.limits.bucket import AdmitReply, allowed_without_limits, new_lease_id, to_result
from gg.limits.config import RateLimitConfig
from gg.limits.lua import ADMIT, FINISH

log = structlog.get_logger("gg.limits.ratelimit")

_REASONS: dict[str, LimitReason] = {
    "ok": "ok",
    "rpm": "rpm",
    "tpm": "tpm",
    "tokens_exceed_limit": "tokens_exceed_limit",
    "concurrency": "concurrency",
}


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def decode_admit(raw: Sequence[object]) -> AdmitReply:
    ints = [int(_text(v)) for v in (raw[0], *raw[2:])]
    return AdmitReply(
        allowed=ints[0] == 1,
        reason=_REASONS[_text(raw[1])],
        rpm_level=ints[1],
        tpm_level=ints[2],
        retry_ms=ints[3],
        in_use=ints[4],
        rpm_reset_ms=ints[5],
        tpm_reset_ms=ints[6],
        now_ms=ints[7],
    )


class RedisRateLimiter:
    """rpm + tpm + concurrency in one atomic admit.lua call; buckets refill on redis TIME"""

    def __init__(
        self,
        redis: Redis,
        cfg: RateLimitConfig,
        *,
        prefix: str = "gg",
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._redis = redis
        self._cfg = cfg
        self._prefix = prefix
        self._admit = redis.register_script(ADMIT)
        self._finish = redis.register_script(FINISH)
        # virtual time for tests only; production leaves it None so the scripts use redis TIME
        self._now_ms = now_ms

    def _keys(self, key_id: str) -> list[str]:
        scope = f"{self._prefix}:rl:{{k:{key_id}}}"
        return [f"{scope}:rpm", f"{scope}:tpm", f"{scope}:conc"]

    def _now_arg(self) -> str:
        return "" if self._now_ms is None else str(self._now_ms())

    async def acquire(
        self, key_id: str, tokens_estimate: int, /, *, limits: BucketLimits, lease_ttl_s: float
    ) -> LimitResult:
        if not limits.enabled:
            return allowed_without_limits(limits)
        lease_id = new_lease_id()
        raw = await self._admit(
            keys=self._keys(key_id),
            args=[
                limits.rpm,
                repr(limits.rpm_capacity),
                limits.tpm,
                repr(limits.tpm_capacity),
                tokens_estimate,
                limits.max_concurrent,
                lease_id,
                int(lease_ttl_s * 1000),
                self._cfg.bucket_ttl_ms,
                self._now_arg(),
            ],
        )
        reply = decode_admit(raw)
        return to_result(reply, limits, key_id=key_id, lease_id=lease_id, need=tokens_estimate, local=False)

    async def finish(self, lease: Lease, actual_tokens: int | None, /) -> None:
        limits = lease.limits
        delta = 0 if actual_tokens is None else actual_tokens - lease.tokens_reserved
        keys = self._keys(lease.key_id)
        await self._finish(
            keys=[keys[1], keys[2]],
            args=[
                limits.tpm,
                repr(limits.tpm_capacity),
                delta,
                lease.lease_id,
                self._cfg.bucket_ttl_ms,
                self._now_arg(),
            ],
        )

    async def load(self) -> list[str]:
        """preloads the scripts (EVALSHA reloads them on NOSCRIPT anyway) and returns their shas"""
        return [str(await self._redis.script_load(script)) for script in (ADMIT, FINISH)]


class ResilientRateLimiter:
    """decorator: redis first; on error or timeout use a local bucket and skip redis for a cooldown"""

    def __init__(
        self,
        primary: RateLimiter,
        fallback: RateLimiter,
        cfg: RateLimitConfig,
        *,
        clock: Clock,
        hooks: LimitsHooks | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._cfg = cfg
        self._clock = clock
        self._hooks = hooks or NullLimitsHooks()
        self._failures = 0
        self._skip_until = 0.0
        self._degraded = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    async def acquire(
        self, key_id: str, tokens_estimate: int, /, *, limits: BucketLimits, lease_ttl_s: float
    ) -> LimitResult:
        if not limits.enabled:
            return allowed_without_limits(limits)
        if self._clock.monotonic() >= self._skip_until:
            try:
                async with asyncio.timeout(self._cfg.redis_timeout_ms / 1000):
                    result = await self._primary.acquire(
                        key_id, tokens_estimate, limits=limits, lease_ttl_s=lease_ttl_s
                    )
            except (RedisError, TimeoutError, OSError) as exc:
                self._failed(exc)
            else:
                self._recovered()
                return result
        if self._cfg.fail_mode == "closed":
            raise RateLimiterUnavailableError("The rate limiter is unavailable.", retry_after_s=5)
        result = await self._fallback.acquire(key_id, tokens_estimate, limits=limits, lease_ttl_s=lease_ttl_s)
        return replace(result, degraded=True)

    async def finish(self, lease: Lease, actual_tokens: int | None, /) -> None:
        if lease.local:
            await self._fallback.finish(lease, actual_tokens)
            return
        try:
            async with asyncio.timeout(self._cfg.redis_timeout_ms / 1000):
                await self._primary.finish(lease, actual_tokens)
        except (RedisError, TimeoutError, OSError) as exc:
            # the lease self-expires and the bucket refills on its own; nothing to undo
            self._hooks.backend_error("finish")
            log.warning("ratelimit.finish_failed", key_id=lease.key_id, error=type(exc).__name__)

    def _failed(self, exc: BaseException) -> None:
        self._hooks.backend_error("acquire")
        self._failures += 1
        # failures only reset on success, so a failed half-open probe reopens the breaker at once
        if self._failures >= self._cfg.breaker_failures:
            self._skip_until = self._clock.monotonic() + self._cfg.breaker_cooldown_s
        if not self._degraded:
            self._degraded = True
            self._hooks.degraded(True)
            log.warning("ratelimit.degraded", error=type(exc).__name__, fail_mode=self._cfg.fail_mode)

    def _recovered(self) -> None:
        self._failures = 0
        if self._degraded:
            self._degraded = False
            self._hooks.degraded(False)
            log.info("ratelimit.recovered")
