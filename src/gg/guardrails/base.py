"""public guardrail contracts; the only module other feature packages may import"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from gg.core.context import ContextKey
from gg.core.guard_types import Finding, Verdict
from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest, Role
from gg.guardrails.vault import GuardVault

type GuardStage = Literal["input", "output"]
type Streaming = Literal["windowed", "buffer", "post_hoc"]
type SegmentKind = Literal["content", "tool_args", "tool_result", "refusal"]
type ErrorKind = Literal["timeout", "exception", "overload"]


class GuardBackendError(Exception):
    """an expected outage of a guard's remote backend; logged as one line, not a traceback"""


type SkipReason = Literal["sampled_out", "not_applicable"]


class Mode(StrEnum):
    ENFORCE = "enforce"
    SHADOW = "shadow"
    OFF = "off"


class OnError(StrEnum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


ON_ERROR_VERDICT: dict[OnError, Verdict] = {
    OnError.ALLOW: Verdict.ALLOW,
    OnError.FLAG: Verdict.FLAG,
    OnError.BLOCK: Verdict.BLOCK,
}


@dataclass(frozen=True, slots=True)
class Decoded:
    """a base64/hex/tag-smuggled payload found in a segment; start/end index the forwarded text"""

    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Segment:
    """one piece of inspectable text plus where it lives in the request (for write-back)"""

    index: int
    role: Role
    kind: SegmentKind
    msg: int
    text: str
    part: int = 0
    tool_call: int = -1
    inspect: str | None = None
    decoded: tuple[Decoded, ...] = ()

    @property
    def view(self) -> str:
        return self.text if self.inspect is None else self.inspect


@dataclass(frozen=True, slots=True)
class Redaction:
    """a span of segment text to replace; continuation is a regex for an unfinished stream match"""

    segment: int
    start: int
    end: int
    label: str
    score: float = 1.0
    continuation: str | None = None


@dataclass(frozen=True, slots=True)
class GuardFinding(Finding):
    """core Finding plus engine bookkeeping; `verdict` is effective, `would_verdict` is what was detected"""

    would_verdict: Verdict = Verdict.ALLOW
    tier: int = 0
    labels: tuple[str, ...] = ()
    redactions: tuple[Redaction, ...] = ()
    replacements: tuple[Segment, ...] = ()
    skipped: SkipReason | None = None

    @property
    def error_closed(self) -> bool:
        return self.error is not None and self.verdict is Verdict.BLOCK


def finding(
    guard: str,
    stage: GuardStage,
    verdict: Verdict = Verdict.ALLOW,
    *,
    score: float | None = None,
    reason: str = "",
    labels: tuple[str, ...] = (),
    redactions: tuple[Redaction, ...] = (),
    replacements: tuple[Segment, ...] = (),
) -> GuardFinding:
    return GuardFinding(
        guard=guard,
        stage=stage,
        verdict=verdict,
        score=score,
        reason=reason,
        would_verdict=verdict,
        labels=labels,
        redactions=redactions,
        replacements=replacements,
    )


@dataclass(frozen=True, slots=True)
class GuardContext:
    stage: GuardStage
    request_id: str
    segments: tuple[Segment, ...]
    request: ChatRequest
    vault: GuardVault
    key: KeyPolicy | None = None
    prior: tuple[GuardFinding, ...] = ()
    is_final: bool = True


class Guardrail(Protocol):
    """a detector; returns a finding for detections and raises only when it could not decide"""

    @property
    def name(self) -> str: ...
    @property
    def stage(self) -> GuardStage: ...
    @property
    def tier(self) -> int: ...
    @property
    def streaming(self) -> Streaming: ...

    async def check(self, gctx: GuardContext, /) -> GuardFinding: ...


@runtime_checkable
class Holdback(Protocol):
    """windowed output guards: trailing chars that might be the start of an unfinished match"""

    def holdback(self, text: str, /) -> int: ...


@runtime_checkable
class Restorer(Protocol):
    """output finalizer that puts vault values back; runs after every detection guard"""

    def restore(self, text: str, vault: GuardVault, /, *, json_escape: bool = False) -> str: ...


@runtime_checkable
class Redacting(Protocol):
    """marker for guards whose redactions protect upstream; shadowing them needs allow_unredacted_upstream"""

    redacts_upstream: bool


@dataclass(frozen=True, slots=True)
class PolicyRef:
    id: str
    version: str
    hash: str

    def header(self) -> str:
        return f"{self.id}@{self.version}+{self.hash}"


class GuardMetrics(Protocol):
    """hooks for the C10 catalogue (#41-44); the composition root may wire a prometheus implementation"""

    def decision(self, stage: GuardStage, guard: str, action: str, mode: str, /) -> None: ...
    def duration(self, stage: GuardStage, guard: str, seconds: float, /) -> None: ...
    def error(self, stage: GuardStage, guard: str, kind: ErrorKind, /) -> None: ...
    def placeholders(self, direction: str, strategy: str, count: int, /) -> None: ...


class NullGuardMetrics:
    def decision(self, stage: GuardStage, guard: str, action: str, mode: str, /) -> None:
        return None

    def duration(self, stage: GuardStage, guard: str, seconds: float, /) -> None:
        return None

    def error(self, stage: GuardStage, guard: str, kind: ErrorKind, /) -> None:
        return None

    def placeholders(self, direction: str, strategy: str, count: int, /) -> None:
        return None


type RequestPredicate = Callable[[ChatRequest], bool]

# effective policy id@version+hash of the request; c8 keys exact-cache entries on it
POLICY_REF = ContextKey[PolicyRef]("guardrails.policy_ref")
