"""per-key budgets: capped ledgers over utc day and month periods, with holds reserved before the call.

budget hashes, the holds zset and open hold strings carry no ttl, so volatile-lru can never evict spend.
"""

import asyncio
import math
import secrets
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gg.core.clock import Clock
from gg.limits.base import (
    MICROS_PER_USD,
    BudgetCaps,
    BudgetExceededError,
    BudgetPeriod,
    BudgetStatus,
    BudgetUnavailableError,
    Hold,
    LimitsHooks,
    Micros,
    NullLimitsHooks,
)
from gg.limits.config import BudgetConfig
from gg.limits.lua import ADJUST, RESERVE, SETTLE

log = structlog.get_logger("gg.limits.budget")

_PERIOD_LABEL: dict[BudgetPeriod, str] = {"day": "daily", "month": "monthly"}


@dataclass(frozen=True, slots=True)
class PeriodSlot:
    period: BudgetPeriod
    key: str
    cap: Micros
    resets_at: datetime


def next_midnight(now: datetime) -> datetime:
    return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


def next_month(now: datetime) -> datetime:
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, 1, tzinfo=UTC)


def usd(micros: Micros) -> str:
    return f"{micros / MICROS_PER_USD:.6f}"


def _short_usd(micros: Micros) -> str:
    return f"${micros / MICROS_PER_USD:.4f}".rstrip("0").rstrip(".")


class _BudgetLedgerBase(ABC):
    def __init__(
        self, cfg: BudgetConfig, *, clock: Clock, prefix: str = "gg", hooks: LimitsHooks | None = None
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._prefix = prefix
        self._hooks = hooks or NullLimitsHooks()

    def scope(self, key_id: str) -> str:
        return f"{self._prefix}:bud:{{k:{key_id}}}"

    def slots(self, key_id: str, caps: BudgetCaps) -> list[PeriodSlot]:
        now = self._clock.now().astimezone(UTC)
        scope = self.scope(key_id)
        slots: list[PeriodSlot] = []
        if caps.daily is not None:
            slots.append(PeriodSlot("day", f"{scope}:d:{now:%Y%m%d}", caps.daily, next_midnight(now)))
        if caps.monthly is not None:
            slots.append(PeriodSlot("month", f"{scope}:m:{now:%Y%m}", caps.monthly, next_month(now)))
        return slots

    @staticmethod
    def status(slots: Sequence[PeriodSlot], used_before: Sequence[Micros], amount: Micros) -> BudgetStatus:
        def ratio(i: int) -> float:
            cap = slots[i].cap
            return math.inf if cap <= 0 else (used_before[i] + amount) / cap

        i = max(range(len(slots)), key=ratio)
        return BudgetStatus(slots[i].period, slots[i].cap, used_before[i] + amount, slots[i].resets_at)

    def exceeded(self, key_id: str, slot: PeriodSlot, used: Micros) -> BudgetExceededError:
        log.info("budget.exceeded", key_id=key_id, period=slot.period, used_micros=used, cap_micros=slot.cap)
        resets = slot.resets_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        message = (
            f"Budget exceeded for key '{key_id}': {_PERIOD_LABEL[slot.period]} cap {_short_usd(slot.cap)}, "
            f"used {_short_usd(used)}. Resets at {resets}."
        )
        return BudgetExceededError(message, status=BudgetStatus(slot.period, slot.cap, used, slot.resets_at))

    def _now_ms(self) -> int:
        return int(self._clock.time() * 1000)

    @staticmethod
    def _require_amount(amount: Micros) -> None:
        if amount < 0:
            raise ValueError(f"budget amounts must not be negative, got {amount} micro-usd")

    @abstractmethod
    async def adjust_hold(self, hold: Hold, amount: Micros, /) -> bool: ...

    async def extend_hold(self, hold: Hold, extra: Micros, /) -> bool:
        self._require_amount(extra)
        return await self.adjust_hold(hold, hold.amount + extra)

    @abstractmethod
    async def settle(self, hold: Hold, actual: Micros, /) -> None: ...

    async def release(self, hold: Hold, /) -> None:
        await self.settle(hold, 0)
        if hold.state == "settled":
            hold.state = "released"


@dataclass(slots=True)
class _StoredHold:
    key_id: str
    amount: Micros
    state: str
    period_keys: tuple[str, ...]
    expires_ms: int


class InMemoryBudgetLedger(_BudgetLedgerBase):
    """process-local ledger with the same semantics as the lua scripts; tests and redis-less dev"""

    def __init__(
        self, cfg: BudgetConfig, *, clock: Clock, prefix: str = "gg", hooks: LimitsHooks | None = None
    ) -> None:
        super().__init__(cfg, clock=clock, prefix=prefix, hooks=hooks)
        self._periods: dict[str, list[int]] = {}
        self._holds: dict[str, _StoredHold] = {}

    def spent(self, period_key: str) -> Micros:
        return self._periods.get(period_key, [0, 0])[0]

    def reserved(self, period_key: str) -> Micros:
        return self._periods.get(period_key, [0, 0])[1]

    def _row(self, period_key: str) -> list[int]:
        return self._periods.setdefault(period_key, [0, 0])

    def _sweep(self, key_id: str, now: int) -> None:
        swept = 0
        for hold_id, stored in list(self._holds.items()):
            if stored.expires_ms > now:
                continue
            if stored.state == "open" and stored.key_id == key_id:
                for pk in stored.period_keys:
                    row = self._row(pk)
                    row[1] -= stored.amount
                    row[0] += stored.amount
                stored.state = "swept"
                stored.expires_ms = now + self._cfg.tombstone_s * 1000
                swept += 1
            elif stored.state != "open":
                del self._holds[hold_id]
        if swept:
            self._hooks.holds_expired(swept)

    async def preauthorize(
        self, key_id: str, amount: Micros, /, *, caps: BudgetCaps, hold_ttl_s: float
    ) -> tuple[Hold, BudgetStatus]:
        self._require_amount(amount)
        slots = self.slots(key_id, caps)
        if not slots:
            raise ValueError("preauthorize needs at least one capped period")
        now = self._now_ms()
        self._sweep(key_id, now)
        used = [sum(self._periods.get(s.key, [0, 0])) for s in slots]
        for slot, before in zip(slots, used, strict=True):
            if before + amount > slot.cap:
                raise self.exceeded(key_id, slot, before)
        for slot in slots:
            self._row(slot.key)[1] += amount
        hold = Hold(
            secrets.token_hex(8), key_id, amount, tuple(s.key for s in slots), tuple(s.cap for s in slots)
        )
        self._holds[hold.hold_id] = _StoredHold(
            key_id, amount, "open", hold.period_keys, now + int(hold_ttl_s * 1000)
        )
        return hold, self.status(slots, used, amount)

    async def adjust_hold(self, hold: Hold, amount: Micros, /) -> bool:
        self._require_amount(amount)
        stored = self._holds.get(hold.hold_id)
        if stored is None or stored.state != "open":
            return False
        diff = amount - stored.amount
        if diff > 0:
            for pk, cap in zip(stored.period_keys, hold.caps, strict=True):
                if sum(self._periods.get(pk, [0, 0])) + diff > cap:
                    return False
        for pk in stored.period_keys:
            self._row(pk)[1] += diff
        stored.amount = hold.amount = amount
        return True

    async def settle(self, hold: Hold, actual: Micros, /) -> None:
        self._require_amount(actual)
        if hold.state != "open":
            return
        stored = self._holds.get(hold.hold_id)
        if stored is None:
            for pk in hold.period_keys:
                self._row(pk)[0] += actual
        elif stored.state == "open":
            for pk in stored.period_keys:
                row = self._row(pk)
                row[1] -= stored.amount
                row[0] += actual
        elif stored.state == "swept":
            for pk in stored.period_keys:
                self._row(pk)[0] += actual - stored.amount
        if stored is not None:
            stored.state = "settled"
            stored.expires_ms = self._now_ms() + self._cfg.tombstone_s * 1000
        hold.state = "settled"


def _int(value: object) -> int:
    return int(value.decode() if isinstance(value, bytes) else str(value))


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class RedisBudgetLedger(_BudgetLedgerBase):
    """capped ledger over reserve/adjust/settle.lua; redis errors on preauthorize fail closed"""

    def __init__(
        self,
        redis: Redis,
        cfg: BudgetConfig,
        *,
        clock: Clock,
        prefix: str = "gg",
        hooks: LimitsHooks | None = None,
        virtual_time: bool = False,
    ) -> None:
        super().__init__(cfg, clock=clock, prefix=prefix, hooks=hooks)
        self._redis = redis
        self._reserve = redis.register_script(RESERVE)
        self._adjust = redis.register_script(ADJUST)
        self._settle = redis.register_script(SETTLE)
        self._timeout_s = cfg.redis_timeout_ms / 1000
        # tests only: hold expiry follows the injected clock instead of redis TIME
        self._virtual_time = virtual_time

    def _now_arg(self) -> str:
        return str(self._now_ms()) if self._virtual_time else ""

    async def load(self) -> list[str]:
        return [str(await self._redis.script_load(script)) for script in (RESERVE, ADJUST, SETTLE)]

    async def preauthorize(
        self, key_id: str, amount: Micros, /, *, caps: BudgetCaps, hold_ttl_s: float
    ) -> tuple[Hold, BudgetStatus]:
        self._require_amount(amount)
        slots = self.slots(key_id, caps)
        if not slots:
            raise ValueError("preauthorize needs at least one capped period")
        hold_id = secrets.token_hex(8)
        scope = self.scope(key_id)
        keys = [*(s.key for s in slots), f"{scope}:holds", f"{scope}:h:{hold_id}"]
        args: list[str | int] = [
            len(slots),
            *(s.cap for s in slots),
            amount,
            hold_id,
            int(hold_ttl_s * 1000),
            self._cfg.sweep_limit,
            f"{scope}:h:",
            self._cfg.tombstone_s * 1000,
            self._now_arg(),
        ]
        try:
            async with asyncio.timeout(self._timeout_s):
                raw = await self._reserve(keys=keys, args=args)
        except (RedisError, TimeoutError, OSError) as exc:
            self._hooks.backend_error("reserve")
            log.warning("budget.unavailable", key_id=key_id, error=type(exc).__name__)
            raise BudgetUnavailableError(
                "The budget service is unavailable; try again shortly.", retry_after_s=5
            ) from exc
        ok, swept, refused, *used_raw = (_int(v) for v in raw)
        used = list(used_raw)
        if swept:
            self._hooks.holds_expired(swept)
        if not ok:
            raise self.exceeded(key_id, slots[refused - 1], used[refused - 1])
        hold = Hold(hold_id, key_id, amount, tuple(s.key for s in slots), tuple(s.cap for s in slots))
        return hold, self.status(slots, used, amount)

    async def adjust_hold(self, hold: Hold, amount: Micros, /) -> bool:
        self._require_amount(amount)
        if hold.state != "open":
            return False
        try:
            async with asyncio.timeout(self._timeout_s):
                raw = await self._adjust(keys=[self._hold_key(hold)], args=[amount, *hold.caps])
        except (RedisError, TimeoutError, OSError) as exc:
            # the hold keeps its old size; the larger of the two is the conservative choice for a shrink
            self._hooks.backend_error("adjust")
            log.warning("budget.adjust_failed", key_id=hold.key_id, error=type(exc).__name__)
            return False
        applied = _int(raw[0]) == 1
        if applied:
            hold.amount = amount
        return applied

    async def settle(self, hold: Hold, actual: Micros, /) -> None:
        self._require_amount(actual)
        if hold.state != "open":
            return
        scope = self.scope(hold.key_id)
        try:
            async with asyncio.timeout(self._timeout_s):
                state = await self._settle(
                    keys=[self._hold_key(hold), f"{scope}:holds", *hold.period_keys],
                    args=[actual, hold.hold_id, self._cfg.tombstone_s * 1000],
                )
        except (RedisError, TimeoutError, OSError) as exc:
            # the open hold is swept later and counted at its reserved amount, which never under-counts
            self._hooks.backend_error("settle")
            log.error(
                "budget.settle_failed", key_id=hold.key_id, actual_micros=actual, error=type(exc).__name__
            )
            return
        if _text(state) == "missing":
            log.warning("budget.settle_missing_hold", key_id=hold.key_id, hold_id=hold.hold_id)
        hold.state = "settled"

    def _hold_key(self, hold: Hold) -> str:
        return f"{self.scope(hold.key_id)}:h:{hold.hold_id}"
