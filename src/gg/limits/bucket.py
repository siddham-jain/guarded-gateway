"""pure-python token bucket: the reference model for admit.lua and the in-process limiter.

every formula mirrors gg.limits.lua operation for operation, so both runtimes make identical decisions.
"""

import math
import secrets
from dataclasses import dataclass, field

from gg.core.clock import Clock
from gg.limits.base import BucketLimits, Lease, LimitReason, LimitResult

CONCURRENCY_RETRY_MS = 1000


@dataclass(slots=True)
class BucketLevel:
    v: float
    ts: int


@dataclass(frozen=True, slots=True)
class AdmitReply:
    """the decoded admit.lua reply; levels are floors, 0 when the limit is off"""

    allowed: bool
    reason: LimitReason
    rpm_level: int
    tpm_level: int
    retry_ms: int
    in_use: int
    rpm_reset_ms: int
    tpm_reset_ms: int
    now_ms: int


def refill(state: BucketLevel | None, now_ms: int, rate: int, cap: float) -> float:
    v = cap if state is None else state.v
    ts = now_ms if state is None else state.ts
    # max(0, ...) tolerates time going backwards
    return min(cap, v + max(0, now_ms - ts) * rate / 60000.0)


def wait_ms(level: float | None, want: float, rate: int) -> int:
    if level is None or level >= want:
        return 0
    return math.ceil((want - level) * 60000.0 / rate)


def reset_ms(level: float | None, rate: int, cap: float) -> int:
    if level is None:
        return 0
    return max(0, math.ceil((cap - level) * 60000.0 / rate))


def reconcile(level: float, delta: int, cap: float) -> float:
    """refund (delta < 0) or debit (delta > 0), clamped to [-capacity, capacity]"""
    return max(-cap, min(cap, level - delta))


def new_lease_id() -> str:
    return secrets.token_hex(8)


def to_result(
    reply: AdmitReply,
    limits: BucketLimits,
    *,
    key_id: str,
    lease_id: str,
    need: int,
    local: bool,
    degraded: bool = False,
) -> LimitResult:
    retry_after: float | None = None
    if not reply.allowed and reply.reason != "tokens_exceed_limit":
        retry_after = reply.retry_ms / 1000
    # rpm-only keys need no finish call: nothing to reconcile or release
    needs_finish = reply.allowed and (limits.tpm > 0 or limits.max_concurrent > 0)
    lease = Lease(key_id, lease_id, need, limits, local=local) if needs_finish else None
    return LimitResult(
        allowed=reply.allowed,
        reason=reply.reason,
        limits=limits,
        remaining_requests=max(0, reply.rpm_level) if limits.rpm else None,
        reset_requests_s=reply.rpm_reset_ms / 1000 if limits.rpm else None,
        remaining_tokens=max(0, reply.tpm_level) if limits.tpm else None,
        reset_tokens_s=reply.tpm_reset_ms / 1000 if limits.tpm else None,
        retry_after_s=retry_after,
        lease=lease,
        degraded=degraded,
    )


def allowed_without_limits(limits: BucketLimits) -> LimitResult:
    return LimitResult(allowed=True, reason="ok", limits=limits)


@dataclass(slots=True)
class KeyState:
    rpm: BucketLevel | None = None
    tpm: BucketLevel | None = None
    leases: dict[str, int] = field(default_factory=lambda: {})


class LocalRateLimiter:
    """in-process limiter: tests, redis-less dev, and the fallback while redis is down"""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._keys: dict[str, KeyState] = {}

    def _now_ms(self) -> int:
        return int(self._clock.time() * 1000)

    async def acquire(
        self, key_id: str, tokens_estimate: int, /, *, limits: BucketLimits, lease_ttl_s: float
    ) -> LimitResult:
        if not limits.enabled:
            return allowed_without_limits(limits)
        lease_id = new_lease_id()
        reply = self.admit(key_id, tokens_estimate, limits, lease_id, int(lease_ttl_s * 1000))
        return to_result(reply, limits, key_id=key_id, lease_id=lease_id, need=tokens_estimate, local=True)

    def admit(self, key_id: str, need: int, limits: BucketLimits, lease_id: str, ttl_ms: int) -> AdmitReply:
        now = self._now_ms()
        state = self._keys.setdefault(key_id, KeyState())
        in_use = 0
        if limits.max_concurrent > 0:
            state.leases = {lid: exp for lid, exp in state.leases.items() if exp > now}
            in_use = len(state.leases)
        rv = refill(state.rpm, now, limits.rpm, limits.rpm_capacity) if limits.rpm > 0 else None
        tv = refill(state.tpm, now, limits.tpm, limits.tpm_capacity) if limits.tpm > 0 else None

        def reply(allowed: bool, reason: LimitReason, retry: int, used: int) -> AdmitReply:
            return AdmitReply(
                allowed=allowed,
                reason=reason,
                rpm_level=math.floor(rv) if rv is not None else 0,
                tpm_level=math.floor(tv) if tv is not None else 0,
                retry_ms=retry,
                in_use=used,
                rpm_reset_ms=reset_ms(rv, limits.rpm, limits.rpm_capacity),
                tpm_reset_ms=reset_ms(tv, limits.tpm, limits.tpm_capacity),
                now_ms=now,
            )

        if tv is not None and need > limits.tpm_capacity:
            return reply(False, "tokens_exceed_limit", 0, in_use)
        if limits.max_concurrent > 0 and in_use >= limits.max_concurrent:
            return reply(False, "concurrency", CONCURRENCY_RETRY_MS, in_use)
        rw, tw = wait_ms(rv, 1, limits.rpm), wait_ms(tv, need, limits.tpm)
        if rw > 0 or tw > 0:
            return reply(False, "rpm" if rw >= tw else "tpm", max(rw, tw), in_use)
        if rv is not None:
            rv -= 1
            state.rpm = BucketLevel(rv, now)
        if tv is not None:
            tv -= need
            state.tpm = BucketLevel(tv, now)
        if limits.max_concurrent > 0:
            state.leases[lease_id] = now + ttl_ms
            in_use += 1
        return reply(True, "ok", 0, in_use)

    async def finish(self, lease: Lease, actual_tokens: int | None, /) -> None:
        state = self._keys.get(lease.key_id)
        if state is None:
            return
        limits = lease.limits
        delta = 0 if actual_tokens is None else actual_tokens - lease.tokens_reserved
        if limits.tpm > 0 and delta != 0:
            now = self._now_ms()
            level = refill(state.tpm, now, limits.tpm, limits.tpm_capacity)
            state.tpm = BucketLevel(reconcile(level, delta, limits.tpm_capacity), now)
        state.leases.pop(lease.lease_id, None)

    def in_use(self, key_id: str) -> int:
        state = self._keys.get(key_id)
        return len(state.leases) if state is not None else 0
