from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, cast

from gg.core.aio import Deadline, run_shielded
from gg.core.cache_types import CacheState, CacheStatus
from gg.core.clock import Clock
from gg.core.deployment import Deployment
from gg.core.guard_types import Finding, OutputVerdict
from gg.core.keypolicy import KeyPolicy
from gg.core.normalize import Normalization
from gg.core.routing_types import RouteDecision
from gg.core.schema import ChatRequest
from gg.core.usage import AttemptRecord, TokenEstimates, UsageRecord
from gg.core.vault import PlaceholderVault

type Outcome = Literal[
    "completed",
    "client_disconnected",
    "rejected",
    "upstream_error",
    "guard_aborted",
    "internal_error",
    "shutdown",
]


class ContextKey[T]:
    """typed key for the extension bag; plugins add state without editing RequestContext"""

    def __init__(self, name: str) -> None:
        self.name = name


class StageTimings:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.marks: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.starts: dict[str, float] = {}

    def mark(self, name: str, at: float | None = None) -> None:
        if name not in self.marks:
            self.marks[name] = self._clock.monotonic() if at is None else at

    def record(self, name: str, seconds: float) -> None:
        self.durations[name] = self.durations.get(name, 0.0) + seconds

    def start(self, name: str, at: float) -> None:
        self.starts.setdefault(name, at)

    @contextmanager
    def measure(self, name: str) -> Generator[None]:
        start = self._clock.monotonic()
        self.start(name, start)
        try:
            yield
        finally:
            self.record(name, self._clock.monotonic() - start)

    def between(self, start: str, end: str) -> float | None:
        if start in self.marks and end in self.marks:
            return self.marks[end] - self.marks[start]
        return None

    def server_timing(self) -> str:
        return ", ".join(f"{name};dur={secs * 1000:.2f}" for name, secs in self.durations.items())


class FinalizerOrder(IntEnum):
    LIMITS = 10
    BUDGET = 20
    CACHE_WRITE = 50
    POST_HOC = 60
    METRICS = 80
    TRACE = 90
    LOG = 100


class Finalizers:
    """end-of-request work run once, shielded and ordered, by the response writer"""

    def __init__(self) -> None:
        self._items: list[tuple[int, int, str, Callable[[], Awaitable[None]]]] = []
        self._ran = False

    def defer(self, name: str, fn: Callable[[], Awaitable[None]], order: int = 100) -> None:
        self._items.append((order, len(self._items), name, fn))

    async def run(self, *, timeout_s: float) -> None:
        if self._ran:
            return
        self._ran = True
        for _, _, name, fn in sorted(self._items, key=lambda item: (item[0], item[1])):
            await run_shielded(fn, name=name, timeout_s=timeout_s)


@dataclass(slots=True, kw_only=True)
class RequestContext:
    request_id: str
    received_at: float
    received_unix: int
    key: KeyPolicy
    original: ChatRequest
    request: ChatRequest
    deadline: Deadline
    timings: StageTimings
    config_hash: str = ""
    trace_id: str | None = None
    vault: PlaceholderVault = field(default_factory=PlaceholderVault)
    route: RouteDecision | None = None
    guard_findings: list[Finding] = field(default_factory=lambda: [])
    scrubbed: ChatRequest | None = None
    cache_status: CacheStatus = "miss"
    cache: CacheState | None = None
    output_verdict: OutputVerdict | None = None
    # the reply in placeholder space (pii not restored), set by the output guard for tracing
    reply_text: str | None = None
    estimates: TokenEstimates | None = None
    usage: UsageRecord | None = None
    attempts: list[AttemptRecord] = field(default_factory=lambda: [])
    served_by: Deployment | None = None
    ignored_params: set[str] = field(default_factory=lambda: set())
    normalizations: tuple[Normalization, ...] = ()
    response_headers: dict[str, str] = field(default_factory=lambda: {})
    finalizers: Finalizers = field(default_factory=Finalizers)
    outcome: Outcome | None = None
    _ext: dict[str, object] = field(default_factory=lambda: {})

    def get[T](self, key: ContextKey[T]) -> T | None:
        return cast("T | None", self._ext.get(key.name))

    def set[T](self, key: ContextKey[T], value: T) -> None:
        self._ext[key.name] = value

    def __repr__(self) -> str:
        return f"RequestContext(request_id={self.request_id!r}, key={self.key.id!r})"
