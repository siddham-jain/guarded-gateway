from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

from gg.core.clock import Clock
from gg.core.deployment import Deployment

type BreakerStateName = Literal["closed", "open", "half_open"]
type CooldownScope = Literal["deployment", "provider"]


@dataclass(frozen=True, slots=True)
class BreakerConfig:
    consecutive_failures: int = 3
    window_s: float = 60.0
    window_calls: int = 20
    min_calls: int = 10
    failure_rate: float = 0.5
    cooldown_s: float = 30.0
    cooldown_max_s: float = 300.0


@dataclass(frozen=True, slots=True)
class Cooldown:
    seconds: float
    reason: str


@dataclass(frozen=True, slots=True)
class BreakerSnapshot:
    state: BreakerStateName
    reason: str | None
    open_for_s: float
    consecutive_failures: int
    failure_rate: float


class BreakerListener(Protocol):
    def breaker_transition(
        self, deployment_id: str, old: BreakerStateName, new: BreakerStateName, reason: str, /
    ) -> None: ...


class Permit:
    """one call's admission; settle exactly once with success, failure or neutral"""

    __slots__ = ("_breaker", "_settled", "probe")

    def __init__(self, breaker: "CircuitBreaker", *, probe: bool) -> None:
        self._breaker = breaker
        self._settled = False
        self.probe = probe

    def _settle(self) -> None:
        if self._settled:
            raise RuntimeError("permit already settled")
        self._settled = True

    def success(self) -> None:
        self._settle()
        self._breaker._state.on_success(self._breaker, self.probe)

    def failure(self) -> None:
        self._settle()
        self._breaker._state.on_failure(self._breaker, self.probe)

    def neutral(self) -> None:
        self._settle()
        self._breaker._state.on_neutral(self._breaker, self.probe)


class _State:
    name: ClassVar[BreakerStateName]

    def acquire(self, b: "CircuitBreaker") -> Permit | None:
        raise NotImplementedError

    def available(self, b: "CircuitBreaker") -> bool:
        raise NotImplementedError

    def on_success(self, b: "CircuitBreaker", probe: bool) -> None:
        return None

    def on_failure(self, b: "CircuitBreaker", probe: bool) -> None:
        return None

    def on_neutral(self, b: "CircuitBreaker", probe: bool) -> None:
        return None


class _Closed(_State):
    name = "closed"

    def acquire(self, b: "CircuitBreaker") -> Permit | None:
        return Permit(b, probe=False)

    def available(self, b: "CircuitBreaker") -> bool:
        return True

    def on_success(self, b: "CircuitBreaker", probe: bool) -> None:
        b._record(ok=True)

    def on_failure(self, b: "CircuitBreaker", probe: bool) -> None:
        if b._record(ok=False):
            b._open(b._cooldown_s, "trip")


class _Open(_State):
    name = "open"

    def __init__(self, until: float, reason: str) -> None:
        self.until = until
        self.reason = reason

    def acquire(self, b: "CircuitBreaker") -> Permit | None:
        if b._clock.monotonic() < self.until:
            return None
        b._to(_HalfOpen(), "cooldown_elapsed")
        return b._state.acquire(b)

    def available(self, b: "CircuitBreaker") -> bool:
        return b._clock.monotonic() >= self.until


class _HalfOpen(_State):
    name = "half_open"

    def __init__(self) -> None:
        self.probe_out = False

    def acquire(self, b: "CircuitBreaker") -> Permit | None:
        if self.probe_out:
            return None
        self.probe_out = True
        return Permit(b, probe=True)

    def available(self, b: "CircuitBreaker") -> bool:
        return not self.probe_out

    def on_success(self, b: "CircuitBreaker", probe: bool) -> None:
        if probe:
            b._cooldown_s = b._config.cooldown_s
            b._reset_window()
            b._to(_Closed(), "probe_ok")

    def on_failure(self, b: "CircuitBreaker", probe: bool) -> None:
        if probe:
            b._cooldown_s = min(2 * b._cooldown_s, b._config.cooldown_max_s)
            b._open(b._cooldown_s, "probe_fail")

    def on_neutral(self, b: "CircuitBreaker", probe: bool) -> None:
        if probe:
            self.probe_out = False


class CircuitBreaker:
    """per-deployment breaker (state pattern); transitions out of open are lazy, on the next acquire"""

    def __init__(
        self,
        deployment_id: str,
        provider: str,
        config: BreakerConfig,
        clock: Clock,
        listener: BreakerListener | None = None,
    ) -> None:
        self.deployment_id = deployment_id
        self.provider = provider
        self._config = config
        self._clock = clock
        self._listener = listener
        self._state: _State = _Closed()
        self._calls: deque[tuple[float, bool]] = deque(maxlen=config.window_calls)
        self._consecutive = 0
        self._cooldown_s = config.cooldown_s

    @property
    def state(self) -> BreakerStateName:
        if isinstance(self._state, _Open) and self._state.available(self):
            return "half_open"
        return self._state.name

    def available(self) -> bool:
        return self._state.available(self)

    def try_acquire(self) -> Permit | None:
        return self._state.acquire(self)

    def retry_in(self) -> float:
        if isinstance(self._state, _Open):
            return max(0.0, self._state.until - self._clock.monotonic())
        return 0.0

    def record_late_failure(self) -> None:
        # post-commit stream failures: the permit was already settled as a success
        self._state.on_failure(self, False)

    def force_open(self, cooldown: Cooldown) -> None:
        until = self._clock.monotonic() + cooldown.seconds
        if isinstance(self._state, _Open) and self._state.until >= until:
            return
        self._to(_Open(until, cooldown.reason), cooldown.reason)

    def snapshot(self) -> BreakerSnapshot:
        failures = sum(1 for _, ok in self._calls if not ok)
        return BreakerSnapshot(
            state=self.state,
            reason=self._state.reason if isinstance(self._state, _Open) else None,
            open_for_s=self.retry_in(),
            consecutive_failures=self._consecutive,
            failure_rate=failures / len(self._calls) if self._calls else 0.0,
        )

    def _record(self, *, ok: bool) -> bool:
        now = self._clock.monotonic()
        self._calls.append((now, ok))
        while self._calls and self._calls[0][0] < now - self._config.window_s:
            self._calls.popleft()
        if ok:
            self._consecutive = 0
            return False
        self._consecutive += 1
        if self._consecutive >= self._config.consecutive_failures:
            return True
        failures = sum(1 for _, call_ok in self._calls if not call_ok)
        return (
            len(self._calls) >= self._config.min_calls
            and failures / len(self._calls) > self._config.failure_rate
        )

    def _reset_window(self) -> None:
        self._calls.clear()
        self._consecutive = 0

    def _open(self, seconds: float, reason: str) -> None:
        self._to(_Open(self._clock.monotonic() + seconds, reason), reason)

    def _to(self, state: _State, reason: str) -> None:
        old = self._state.name
        self._state = state
        if self._listener is not None:
            self._listener.breaker_transition(self.deployment_id, old, state.name, reason)


class BreakerRegistry:
    """in-process breakers by deployment id; provider cooldowns also reach breakers created later"""

    def __init__(
        self,
        clock: Clock,
        config: BreakerConfig | None = None,
        listener: BreakerListener | None = None,
    ) -> None:
        self._clock = clock
        self._config = config or BreakerConfig()
        self._listener = listener
        self._breakers: dict[str, CircuitBreaker] = {}
        self._provider_open: dict[str, tuple[float, str]] = {}

    def get(self, deployment: Deployment) -> CircuitBreaker:
        breaker = self._breakers.get(deployment.id)
        if breaker is None:
            breaker = CircuitBreaker(
                deployment.id, deployment.provider, self._config, self._clock, self._listener
            )
            self._breakers[deployment.id] = breaker
            pending = self._provider_open.get(deployment.provider)
            if pending is not None and pending[0] > self._clock.monotonic():
                breaker.force_open(Cooldown(pending[0] - self._clock.monotonic(), pending[1]))
        return breaker

    def force_open(
        self, deployment: Deployment, cooldown: Cooldown, *, scope: CooldownScope = "deployment"
    ) -> None:
        if scope == "provider":
            until = self._clock.monotonic() + cooldown.seconds
            current = self._provider_open.get(deployment.provider)
            if current is None or current[0] < until:
                self._provider_open[deployment.provider] = (until, cooldown.reason)
            for breaker in self._breakers.values():
                if breaker.provider == deployment.provider:
                    breaker.force_open(cooldown)
        self.get(deployment).force_open(cooldown)

    def snapshot(self) -> dict[str, BreakerSnapshot]:
        return {dep_id: breaker.snapshot() for dep_id, breaker in self._breakers.items()}
