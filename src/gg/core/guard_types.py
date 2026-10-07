from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class Verdict(IntEnum):
    ALLOW = 0
    FLAG = 1
    REDACT = 2
    BLOCK = 3


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class Finding:
    guard: str
    stage: Literal["input", "output"]
    verdict: Verdict
    score: float | None
    reason: str
    spans: tuple[Span, ...] = ()
    replacement: str | None = None
    latency_ms: float = 0.0
    error: str | None = None
    mode: Literal["enforce", "shadow"] = "enforce"


@dataclass(frozen=True, slots=True)
class OutputVerdict:
    verdict: Verdict
    findings: tuple[Finding, ...] = ()
    post_hoc_pending: bool = False
