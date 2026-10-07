import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import httpx2
import structlog

from gg import __version__
from gg.core.clock import Clock
from gg.core.jsonutil import loads
from gg.routing.base import FallbackReason
from gg.routing.config import JevBreakerConfig, JevConfig

log = structlog.get_logger("gg.routing.jev")

type Sleep = Callable[[float], Awaitable[None]]
type BreakerState = Literal["closed", "open", "half_open"]


class JevBreaker:
    """opens after n consecutive health failures, or at once on auth errors; half-open admits one probe"""

    def __init__(self, cfg: JevBreakerConfig, clock: Clock) -> None:
        self._cfg = cfg
        self._clock = clock
        self._consecutive = 0
        self._open_until: float | None = None
        self._probe_out = False

    @property
    def state(self) -> BreakerState:
        if self._open_until is None:
            return "closed"
        return "open" if self._clock.monotonic() < self._open_until else "half_open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "open" or self._probe_out:
            return False
        self._probe_out = True
        return True

    def success(self) -> None:
        self._consecutive = 0
        self._open_until = None
        self._probe_out = False

    def failure(self) -> None:
        self._consecutive += 1
        if self._probe_out or self._consecutive >= self._cfg.consecutive_failures:
            self.trip(self._cfg.cooldown_s)

    def neutral(self) -> None:
        self._probe_out = False

    def trip(self, seconds: float) -> None:
        self._open_until = self._clock.monotonic() + seconds
        self._probe_out = False


@dataclass(frozen=True, slots=True)
class JevAnswer:
    body: Mapping[str, Any]
    latency_ms: float
    request_id: str | None


@dataclass(frozen=True, slots=True)
class JevFailure:
    reason: FallbackReason
    latency_ms: float
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Attempt:
    """one http attempt: either a parsed 200 body or a failure with its retry hint"""

    body: Mapping[str, Any] | None
    reason: FallbackReason | None
    retryable: bool = False
    retry_after_s: float | None = None
    request_id: str | None = None


class JevClient:
    """raw httpx2 client for /v1/systemone: per-attempt timeout, total deadline, at most one retry"""

    def __init__(
        self,
        http: httpx2.AsyncClient,
        api_key: str,
        cfg: JevConfig,
        clock: Clock,
        *,
        breaker: JevBreaker | None = None,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._http = http
        self._cfg = cfg
        self._clock = clock
        self.breaker = breaker or JevBreaker(cfg.breaker, clock)
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311
        self._headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": f"gg/{__version__}",
        }

    async def ask(self, body: bytes, deadline_s: float) -> JevAnswer | JevFailure:
        start = self._clock.monotonic()
        deadline = start + deadline_s
        last: _Attempt | None = None
        for attempt in range(1, self._cfg.max_attempts + 1):
            if attempt > 1:
                if last is None or not last.retryable:
                    break
                delay = self._backoff_s(last.retry_after_s)
                if deadline - self._clock.monotonic() - delay < self._cfg.min_attempt_budget_ms / 1000:
                    break
                await self._sleep(delay)
            if not self.breaker.allow():
                if last is None:
                    return JevFailure("circuit_open", 0.0)
                break
            remaining = deadline - self._clock.monotonic()
            last = await self._attempt(body, min(self._cfg.attempt_timeout_ms / 1000, remaining))
            if last.body is not None:
                return JevAnswer(last.body, _ms_since(self._clock, start), last.request_id)
        assert last is not None
        assert last.reason is not None
        return JevFailure(last.reason, _ms_since(self._clock, start), last.request_id)

    async def _attempt(self, body: bytes, timeout_s: float) -> _Attempt:
        timeout = httpx2.Timeout(
            connect=min(self._cfg.connect_timeout_ms / 1000, timeout_s), read=timeout_s, write=0.1, pool=0.05
        )
        try:
            # httpx read timeouts do not bound a trickling body; the outer timeout does
            async with asyncio.timeout(timeout_s):
                resp = await self._http.post(
                    self._cfg.url, content=body, headers=self._headers, timeout=timeout
                )
        except (TimeoutError, httpx2.TimeoutException):
            self.breaker.failure()
            return _Attempt(None, "timeout")
        except httpx2.TransportError as exc:
            self.breaker.failure()
            log.warning("jev.network_error", error=type(exc).__name__)
            return _Attempt(None, "network", retryable=True)
        except asyncio.CancelledError:
            self.breaker.neutral()
            raise
        return self._classify(resp)

    def _classify(self, resp: httpx2.Response) -> _Attempt:
        status = resp.status_code
        request_id = resp.headers.get("x-typesafe-request-id")
        if status == 200:
            try:
                body = loads(resp.content)
            except ValueError:
                body = None
            if not isinstance(body, dict):
                self.breaker.neutral()
                log.warning("jev.parse_error", request_id=request_id)
                return _Attempt(None, "parse_error", request_id=request_id)
            self.breaker.success()
            return _Attempt(body, None, request_id=request_id)
        retryable = status in self._cfg.retry_on_status
        if status in (401, 402, 403) and "html" not in resp.headers.get("content-type", ""):
            self.breaker.trip(self._cfg.breaker.auth_cooldown_s)
            log.error("jev.auth_error", status=status, request_id=request_id)
            return _Attempt(None, "auth", request_id=request_id)
        if status == 429:
            # rate limiting says nothing about jev's health
            self.breaker.neutral()
            return _Attempt(None, "http_429", retryable, _retry_after_s(resp.headers), request_id)
        if status >= 500:
            self.breaker.failure()
            reason: FallbackReason = "http_529" if status == 529 else "http_5xx"
            return _Attempt(None, reason, retryable, _retry_after_s(resp.headers), request_id)
        self.breaker.neutral()
        # bodies are not logged: validation details can echo the scrubbed prompt
        log.warning("jev.client_error", status=status, request_id=request_id)
        return _Attempt(None, "firewall_403" if status == 403 else "http_4xx", request_id=request_id)

    def _backoff_s(self, retry_after_s: float | None) -> float:
        if retry_after_s is not None:
            return retry_after_s
        low, high = self._cfg.retry_backoff_ms
        return self._rng.uniform(low, high) / 1000


def _retry_after_s(headers: httpx2.Headers) -> float | None:
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            return max(0.0, float(value) / scale)
        except ValueError:
            continue
    return None


def _ms_since(clock: Clock, start: float) -> float:
    return round((clock.monotonic() - start) * 1000, 1)
