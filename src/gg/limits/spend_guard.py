import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Annotated, Literal

import structlog
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gg.core.clock import Clock
from gg.limits.base import (
    MICROS_PER_USD,
    LimitsTelemetry,
    Micros,
    NullLimitsTelemetry,
    SpendDenial,
    SpendPeriod,
)

log = structlog.get_logger("gg.limits.spend_guard")

type CapTable = dict[str, Decimal]

DEFAULT_PROVIDER = "*"


def parse_caps(raw: object) -> object:
    """'openai=1.00,gemini=0.50' or a bare '0.25' (applies to every provider) -> {provider: usd}"""
    if not isinstance(raw, str):
        return raw
    table: CapTable = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        name, sep, value = entry.rpartition("=")
        provider = name.strip() if sep else DEFAULT_PROVIDER
        if not provider:
            raise ValueError(f"spend cap entry {entry!r} has an empty provider name")
        try:
            usd = Decimal(value.strip())
        except InvalidOperation:
            raise ValueError(f"spend cap entry {entry!r} is not a decimal usd amount") from None
        table[provider] = usd
    return table


class SpendGuardSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GG_SPEND_", env_file=".env", extra="ignore", frozen=True)

    cap_daily_usd: Annotated[CapTable, NoDecode] = {
        "openai": Decimal("1.00"),
        "anthropic": Decimal("1.00"),
        "gemini": Decimal("0.50"),
    }
    cap_total_usd: Annotated[CapTable, NoDecode] = {
        "openai": Decimal("6.00"),
        "anthropic": Decimal("6.00"),
        "gemini": Decimal("3.00"),
    }
    cap_run_usd: Annotated[CapTable, NoDecode] = {DEFAULT_PROVIDER: Decimal("0.25")}
    guard_fail_mode: Literal["closed", "open"] = "closed"
    # the run cap only applies while GG_RUN_ID is set (ci nightly live evals)
    run_id: str | None = Field(default=None, validation_alias=AliasChoices("GG_RUN_ID", "run_id"))
    redis_timeout_s: float = Field(default=0.2, gt=0)

    @field_validator("cap_daily_usd", "cap_total_usd", "cap_run_usd", mode="before")
    @classmethod
    def _parse(cls, value: object) -> object:
        return parse_caps(value)

    @field_validator("cap_daily_usd", "cap_total_usd", "cap_run_usd")
    @classmethod
    def _non_negative(cls, value: CapTable) -> CapTable:
        for provider, usd in value.items():
            if usd < 0:
                raise ValueError(f"spend cap for {provider!r} must not be negative")
        return value

    def cap_micros(self, period: SpendPeriod, provider: str) -> Micros:
        """unknown providers get a $0 cap, so a new provider can't spend until it is configured"""
        table = {"day": self.cap_daily_usd, "total": self.cap_total_usd, "run": self.cap_run_usd}[period]
        usd = table.get(provider, table.get(DEFAULT_PROVIDER, Decimal(0)))
        return int((usd * MICROS_PER_USD).to_integral_value(rounding=ROUND_FLOOR))


class _CappedSpendGuard:
    """shared cap logic; subclasses only read and increment per-period counters"""

    def __init__(
        self, settings: SpendGuardSettings, *, clock: Clock, telemetry: LimitsTelemetry | None = None
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._telemetry = telemetry or NullLimitsTelemetry()

    def period_keys(self, provider: str, now: datetime) -> list[tuple[SpendPeriod, str]]:
        scope = f"gg:spend:{{p:{provider}}}"
        keys: list[tuple[SpendPeriod, str]] = [
            ("day", f"{scope}:d:{now.astimezone(UTC):%Y%m%d}"),
            ("total", f"{scope}:total"),
        ]
        if self._settings.run_id:
            keys.append(("run", f"{scope}:run:{self._settings.run_id}"))
        return keys

    def _evaluate(
        self, provider: str, periods: Sequence[SpendPeriod], spent: Sequence[Micros], now: datetime
    ) -> SpendDenial | None:
        for period, used in zip(periods, spent, strict=True):
            cap = self._settings.cap_micros(period, provider)
            if used >= cap:
                return self._deny(SpendDenial(provider, period, spent=used, cap=cap, checked_at=now))
        return None

    def _deny(self, denial: SpendDenial) -> SpendDenial:
        self._telemetry.spend_denied(denial.provider, denial.reason)
        log.warning(
            "spend_guard.denied",
            provider=denial.provider,
            reason=denial.reason,
            spent_micros=denial.spent,
            cap_micros=denial.cap,
        )
        return denial

    @staticmethod
    def _validate_amount(micro_usd: Micros) -> None:
        if micro_usd < 0:
            raise ValueError(f"spend must not be negative, got {micro_usd} micro-usd")


class InMemorySpendGuard(_CappedSpendGuard):
    """process-local guard for tests and redis-less dev runs"""

    def __init__(
        self, settings: SpendGuardSettings, *, clock: Clock, telemetry: LimitsTelemetry | None = None
    ) -> None:
        super().__init__(settings, clock=clock, telemetry=telemetry)
        self._spent: dict[str, Micros] = {}

    @property
    def counters(self) -> Mapping[str, Micros]:
        return dict(self._spent)

    async def check(self, provider: str) -> SpendDenial | None:
        now = self._clock.now()
        keys = self.period_keys(provider, now)
        spent = [self._spent.get(key, 0) for _, key in keys]
        return self._evaluate(provider, [period for period, _ in keys], spent, now)

    async def record(self, provider: str, micro_usd: Micros) -> None:
        self._validate_amount(micro_usd)
        if micro_usd == 0:
            return
        for _, key in self.period_keys(provider, self._clock.now()):
            self._spent[key] = self._spent.get(key, 0) + micro_usd


class RedisSpendGuard(_CappedSpendGuard):
    """integer micro-usd counters via INCRBY; no ttl on any key so volatile-lru can never evict spend"""

    def __init__(
        self,
        redis: Redis,
        settings: SpendGuardSettings,
        *,
        clock: Clock,
        telemetry: LimitsTelemetry | None = None,
    ) -> None:
        super().__init__(settings, clock=clock, telemetry=telemetry)
        self._redis = redis
        if settings.guard_fail_mode == "open":
            log.warning("spend_guard.fail_open", detail="redis errors will allow upstream spend")

    async def check(self, provider: str) -> SpendDenial | None:
        now = self._clock.now()
        keys = self.period_keys(provider, now)
        try:
            async with asyncio.timeout(self._settings.redis_timeout_s):
                raw = await self._redis.mget([key for _, key in keys])
        except (RedisError, TimeoutError) as exc:
            return self._unavailable(provider, now, exc)
        spent = [int(value) if value is not None else 0 for value in raw]
        return self._evaluate(provider, [period for period, _ in keys], spent, now)

    async def record(self, provider: str, micro_usd: Micros) -> None:
        self._validate_amount(micro_usd)
        if micro_usd == 0:
            return
        keys = self.period_keys(provider, self._clock.now())
        try:
            async with asyncio.timeout(self._settings.redis_timeout_s):
                async with self._redis.pipeline(transaction=True) as pipe:
                    for _, key in keys:
                        pipe.incrby(key, micro_usd)
                    await pipe.execute()
        except (RedisError, TimeoutError) as exc:
            # the attempt already happened; losing the increment under-counts, so make it loud
            log.error(
                "spend_guard.record_failed", provider=provider, micro_usd=micro_usd, error=type(exc).__name__
            )

    def _unavailable(self, provider: str, now: datetime, exc: Exception) -> SpendDenial | None:
        if self._settings.guard_fail_mode == "open":
            log.warning("spend_guard.unavailable_fail_open", provider=provider, error=type(exc).__name__)
            return None
        return self._deny(SpendDenial(provider, "unavailable", checked_at=now))
