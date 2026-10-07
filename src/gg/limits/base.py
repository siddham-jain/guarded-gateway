from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Protocol

from gg.core.context import ContextKey
from gg.core.errors import QuotaExceededError, RateLimitedError, ServiceUnavailableError

type Micros = int
type SpendPeriod = Literal["day", "total", "run"]
type DenialReason = Literal["day", "total", "run", "unavailable"]

MICROS_PER_USD = 1_000_000


class LimitsTelemetry(Protocol):
    """metric hooks the limits layer calls; gg.observability.Metrics implements it structurally"""

    def pricing_missing(self, provider: str, deployment: str, /) -> None: ...
    def spend_denied(self, provider: str, reason: DenialReason, /) -> None: ...


class NullLimitsTelemetry:
    def pricing_missing(self, provider: str, deployment: str, /) -> None:
        return None

    def spend_denied(self, provider: str, reason: DenialReason, /) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SpendDenial:
    provider: str
    reason: DenialReason
    spent: Micros | None = None
    cap: Micros | None = None
    checked_at: datetime | None = None

    def to_error(self) -> "SpendCapReachedError":
        retry_after: float | None = None
        if self.reason == "day" and self.checked_at is not None:
            midnight = (self.checked_at + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            retry_after = (midnight - self.checked_at).total_seconds()
        if self.reason == "unavailable":
            message = f"Spend guard for provider '{self.provider}' is unavailable."
        else:
            message = f"Provider '{self.provider}' reached its {self.reason} spend cap."
        return SpendCapReachedError(message, retry_after_s=retry_after)


class SpendCapReachedError(ServiceUnavailableError):
    default_code = "provider_spend_cap"


class ProviderSpendGuard(Protocol):
    """gateway-wide spend cap per provider (master §6); callers skip it for unbilled deployments"""

    async def check(self, provider: str) -> SpendDenial | None: ...
    async def record(self, provider: str, micro_usd: Micros) -> None: ...


type LimitReason = Literal["ok", "rpm", "tpm", "tokens_exceed_limit", "concurrency"]
type RejectionLimit = Literal[
    "rpm", "tpm", "tokens_exceed_limit", "concurrency", "budget", "budget_unavailable"
]
type BudgetEvent = Literal["soft_limit", "hard_limit", "fail_closed", "fail_open"]
type BackendOp = Literal["acquire", "finish", "reserve", "adjust", "settle"]
type BudgetPeriod = Literal["day", "month"]
type HoldState = Literal["open", "settled", "released"]


class LimitsHooks(Protocol):
    """metrics seam for c9 (gg_ratelimit_*, gg_budget_*); implementations must be cheap and must not raise"""

    def rejected(self, limit: RejectionLimit, /) -> None: ...
    def budget_event(self, event: BudgetEvent, /) -> None: ...
    def backend_error(self, op: BackendOp, /) -> None: ...
    def degraded(self, active: bool, /) -> None: ...
    def holds_expired(self, count: int, /) -> None: ...


class NullLimitsHooks:
    def rejected(self, limit: RejectionLimit, /) -> None:
        return None

    def budget_event(self, event: BudgetEvent, /) -> None:
        return None

    def backend_error(self, op: BackendOp, /) -> None:
        return None

    def degraded(self, active: bool, /) -> None:
        return None

    def holds_expired(self, count: int, /) -> None:
        return None


@dataclass(frozen=True, slots=True)
class BucketLimits:
    """per-key limits in bucket form; 0 turns a limit off"""

    rpm: int = 0
    rpm_capacity: float = 0.0
    tpm: int = 0
    tpm_capacity: float = 0.0
    max_concurrent: int = 0

    @property
    def enabled(self) -> bool:
        return self.rpm > 0 or self.tpm > 0 or self.max_concurrent > 0


@dataclass(frozen=True, slots=True)
class Lease:
    key_id: str
    lease_id: str
    tokens_reserved: int
    limits: BucketLimits
    local: bool = False


@dataclass(frozen=True, slots=True)
class LimitResult:
    allowed: bool
    reason: LimitReason
    limits: BucketLimits
    remaining_requests: int | None = None
    reset_requests_s: float | None = None
    remaining_tokens: int | None = None
    reset_tokens_s: float | None = None
    retry_after_s: float | None = None
    lease: Lease | None = None
    degraded: bool = False


class RateLimiter(Protocol):
    """master §3.2, extended: limits travel with the call because keys hot-reload"""

    async def acquire(
        self, key_id: str, tokens_estimate: int, /, *, limits: BucketLimits, lease_ttl_s: float
    ) -> LimitResult: ...

    async def finish(self, lease: Lease, actual_tokens: int | None, /) -> None: ...


@dataclass(frozen=True, slots=True)
class BudgetCaps:
    daily: Micros | None = None
    monthly: Micros | None = None

    @property
    def enabled(self) -> bool:
        return self.daily is not None or self.monthly is not None


@dataclass(slots=True)
class Hold:
    hold_id: str
    key_id: str
    amount: Micros
    period_keys: tuple[str, ...]
    caps: tuple[Micros, ...]
    state: HoldState = "open"


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    """the tightest capped period after this reservation"""

    period: BudgetPeriod
    cap: Micros
    used: Micros
    resets_at: datetime

    @property
    def remaining(self) -> Micros:
        return max(0, self.cap - self.used)

    @property
    def ratio(self) -> float:
        return 1.0 if self.cap <= 0 else self.used / self.cap


class BudgetLedger(Protocol):
    """master §3.2 in integer micro-usd; raises BudgetExceededError or BudgetUnavailableError"""

    async def preauthorize(
        self, key_id: str, amount: Micros, /, *, caps: BudgetCaps, hold_ttl_s: float
    ) -> tuple[Hold, BudgetStatus]: ...

    async def adjust_hold(self, hold: Hold, amount: Micros, /) -> bool: ...
    async def extend_hold(self, hold: Hold, extra: Micros, /) -> bool: ...
    async def settle(self, hold: Hold, actual: Micros, /) -> None: ...
    async def release(self, hold: Hold, /) -> None: ...


class TokensExceedLimitError(RateLimitedError):
    default_code = "tokens_exceed_limit"
    client_should_retry = False


class BudgetExceededError(QuotaExceededError):
    def __init__(self, message: str, *, status: BudgetStatus, headers: dict[str, str] | None = None) -> None:
        super().__init__(message, headers=headers)
        self.budget = status


class BudgetUnavailableError(ServiceUnavailableError):
    default_code = "budget_unavailable"


class RateLimiterUnavailableError(ServiceUnavailableError):
    default_code = "ratelimit_unavailable"


@dataclass(slots=True)
class LimitsState:
    """what admission decided, for later stages: c5 shrinks the hold, c6/c7 extend it for a re-ask"""

    result: LimitResult
    hold: Hold | None = None
    budget: BudgetStatus | None = None


LIMITS_STATE: ContextKey[LimitsState] = ContextKey("gg.limits")
