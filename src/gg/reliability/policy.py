import random
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from gg.core.clock import Clock
from gg.core.errors import ProviderError
from gg.reliability.breaker import Cooldown

type Action = Literal["retry", "fallback", "fallback_larger_context", "fail"]

# local or request-specific faults that say nothing about the deployment's health
_NEUTRAL_RETRYABLE_CODES = frozenset({"pool_timeout", "conflict", "malformed"})
_BILLING_CODES = frozenset({"billing", "spend_limit"})


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_retries: int = 1
    max_retries_by_code: Mapping[str, int] = field(
        default_factory=lambda: {"connect_error": 2, "ttft_timeout": 0}
    )
    backoff_base_s: float = 0.25
    backoff_cap_s: float = 4.0
    max_retry_after_wait_s: float = 5.0
    retry_after_jitter_s: float = 0.25
    min_attempt_s: float = 1.5
    budget_ratio: float = 0.10
    budget_min_per_window: int = 3
    budget_window_s: float = 60.0
    max_hops: int = 3
    fallback_on_content_filter: bool = False
    auth_cooldown_s: float = 600.0
    billing_cooldown_s: float = 3600.0
    not_found_cooldown_s: float = 600.0
    rate_limit_cooldown_max_s: float = 60.0


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    delay_s: float = 0.0
    cooldown: Cooldown | None = None


class Backoff(Protocol):
    def delay(self, retry_number: int, /) -> float: ...


class FullJitter:
    """aws full jitter: uniform(0, min(cap, base * 2^n))"""

    def __init__(self, base_s: float, cap_s: float, rng: random.Random) -> None:
        self._base_s = base_s
        self._cap_s = cap_s
        self._rng = rng

    def delay(self, retry_number: int, /) -> float:
        return self._rng.uniform(0.0, min(self._cap_s, self._base_s * 2**retry_number))


@dataclass(slots=True)
class _Bucket:
    index: int
    requests: int = 0
    retries: int = 0


class RetryBudget:
    """retries per sliding window stay under max(floor, ratio * requests); bucketed so memory is constant"""

    def __init__(
        self, *, ratio: float, min_per_window: int, window_s: float, clock: Clock, buckets: int = 6
    ) -> None:
        self._ratio = ratio
        self._min = min_per_window
        self._width = window_s / buckets
        self._buckets: deque[_Bucket] = deque()
        self._count = buckets
        self._clock = clock

    def _current(self) -> _Bucket:
        index = int(self._clock.monotonic() // self._width)
        while self._buckets and self._buckets[0].index <= index - self._count:
            self._buckets.popleft()
        if not self._buckets or self._buckets[-1].index != index:
            self._buckets.append(_Bucket(index))
        return self._buckets[-1]

    def on_request(self) -> None:
        self._current().requests += 1

    def try_spend(self) -> bool:
        current = self._current()
        requests = sum(b.requests for b in self._buckets)
        retries = sum(b.retries for b in self._buckets)
        if retries >= max(self._min, self._ratio * requests):
            return False
        current.retries += 1
        return True


def counts_against_breaker(error: ProviderError) -> bool:
    # auth / quota_day / model_not_found open the breaker through a forced cooldown instead
    if error.kind == "retryable":
        return error.code not in _NEUTRAL_RETRYABLE_CODES
    return error.kind == "fallback" and error.code == "overloaded"


class RetryPolicy:
    """decides retry / fallback / fail from ProviderError.kind, never from the raw http status"""

    def __init__(
        self,
        clock: Clock,
        config: RetryConfig | None = None,
        *,
        backoff: Backoff | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or RetryConfig()
        self._clock = clock
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not crypto
        self._backoff = backoff or FullJitter(
            self.config.backoff_base_s, self.config.backoff_cap_s, self._rng
        )
        self._budgets: dict[str, RetryBudget] = {}

    def budget(self, deployment_id: str) -> RetryBudget:
        budget = self._budgets.get(deployment_id)
        if budget is None:
            cfg = self.config
            budget = RetryBudget(
                ratio=cfg.budget_ratio,
                min_per_window=cfg.budget_min_per_window,
                window_s=cfg.budget_window_s,
                clock=self._clock,
            )
            self._budgets[deployment_id] = budget
        return budget

    def on_attempt(self, deployment_id: str) -> None:
        self.budget(deployment_id).on_request()

    def decide(self, error: ProviderError, deployment_id: str, retries: int, remaining_s: float) -> Decision:
        cfg = self.config
        match error.kind:
            case "client":
                return Decision("fail")
            case "content_filter":
                return Decision("fallback" if cfg.fallback_on_content_filter else "fail")
            case "auth":
                billing = error.code in _BILLING_CODES
                seconds = cfg.billing_cooldown_s if billing else cfg.auth_cooldown_s
                return Decision("fallback", cooldown=Cooldown(seconds, error.code or "auth"))
            case "quota_day":
                seconds = cfg.billing_cooldown_s
                if error.quota_reset_at is not None:
                    seconds = max(0.0, (error.quota_reset_at - self._clock.now()).total_seconds())
                return Decision("fallback", cooldown=Cooldown(seconds, error.code or "per_day"))
            case "fallback":
                if error.code == "context_length":
                    return Decision("fallback_larger_context")
                if error.code == "model_not_found":
                    return Decision(
                        "fallback", cooldown=Cooldown(cfg.not_found_cooldown_s, "model_not_found")
                    )
                return Decision("fallback")
            case "retryable" | "quota_minute":
                return self._decide_retry(error, deployment_id, retries, remaining_s)

    def _decide_retry(
        self, error: ProviderError, deployment_id: str, retries: int, remaining_s: float
    ) -> Decision:
        cfg = self.config
        room = remaining_s - cfg.min_attempt_s
        if error.retry_after_s is not None:
            wait = error.retry_after_s + self._rng.uniform(0.0, cfg.retry_after_jitter_s)
            if wait > min(cfg.max_retry_after_wait_s, room):
                cooldown = None
                if error.retry_after_s > cfg.max_retry_after_wait_s:
                    seconds = min(error.retry_after_s, cfg.rate_limit_cooldown_max_s)
                    cooldown = Cooldown(seconds, "rate_limited")
                return Decision("fallback", cooldown=cooldown)
            delay = wait
        elif error.kind == "quota_minute":
            # a 429 without retry-after gives nothing to wait on; go elsewhere
            return Decision("fallback")
        else:
            delay = self._backoff.delay(retries)
        max_retries = cfg.max_retries_by_code.get(error.code or "", cfg.max_retries)
        if retries >= max_retries or delay > room or not self.budget(deployment_id).try_spend():
            return Decision("fallback")
        return Decision("retry", delay)
