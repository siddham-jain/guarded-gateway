import asyncio
import shutil
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from gg.core.clock import SystemClock
from gg.core.guard_types import Verdict
from gg.core.schema import ChatChunk, ChatRequest, ChunkChoice, Delta
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Segment, Streaming, finding
from gg.guardrails.builtin import default_registry
from gg.guardrails.engine import GuardrailEngine
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.policy.loader import PolicySet, load_policies
from gg.guardrails.registry import GuardDeps, GuardRegistry
from gg.guardrails.rules import RulePackStore
from gg.guardrails.vault import GuardVault
from tests.conftest import make_key, make_request

ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = ROOT / "config" / "policies"
DEFAULT_POLICY: dict[str, Any] = yaml.safe_load((POLICY_DIR / "default.yaml").read_text())


def deps(policy_dir: Path = POLICY_DIR) -> GuardDeps:
    return GuardDeps(packs=RulePackStore(policy_dir), clock=SystemClock())


@cache
def _repo_policies() -> PolicySet:
    return load_policies(POLICY_DIR, default_registry(), deps())


def policy_set(
    policy_dir: Path = POLICY_DIR, registry: GuardRegistry | None = None, key_policy_ids: Sequence[str] = ()
) -> PolicySet:
    if policy_dir == POLICY_DIR and registry is None and not key_policy_ids:
        return _repo_policies()
    return load_policies(
        policy_dir, registry or default_registry(), deps(policy_dir), key_policy_ids=key_policy_ids
    )


def write_policy(tmp: Path, doc: dict[str, Any], name: str = "default.yaml") -> Path:
    """a policy dir with the repo rule packs next to the given policy document"""
    if not (tmp / "rules").exists():
        shutil.copytree(POLICY_DIR / "rules", tmp / "rules")
    (tmp / name).write_text(yaml.safe_dump(doc, sort_keys=False))
    return tmp


def policy_doc(**changes: Any) -> dict[str, Any]:
    """a deep copy of default.yaml with top-level keys replaced"""
    doc: dict[str, Any] = yaml.safe_load(yaml.safe_dump(DEFAULT_POLICY))
    doc.update(changes)
    return doc


def effective(
    request: ChatRequest | None = None, policies: PolicySet | None = None, **key: Any
) -> EffectivePolicy:
    return (policies or policy_set()).effective(make_key(**key), request or make_request())


def engine() -> GuardrailEngine:
    return GuardrailEngine(clock=SystemClock())


def gctx(
    *texts: str, stage: GuardStage = "input", role: Any = "user", vault: GuardVault | None = None
) -> GuardContext:
    segments = tuple(Segment(index=i, role=role, kind="content", msg=i, text=t) for i, t in enumerate(texts))
    return GuardContext(
        stage=stage,
        request_id="req_test",
        segments=segments,
        request=make_request(),
        vault=vault or GuardVault(),
    )


@dataclass
class FakeGuard:
    """scripted guard: verdict, latency, exception; records calls and cancellation"""

    name: str = "fake"
    tier: int = 1
    verdict: Verdict = Verdict.ALLOW
    delay: float = 0.0
    error: BaseException | None = None
    stage: GuardStage = "input"
    streaming: Streaming = "windowed"
    replacements: tuple[Segment, ...] = ()
    calls: int = 0
    cancelled: bool = False
    finished: bool = False

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        self.calls += 1
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        self.finished = True
        return finding(self.name, self.stage, self.verdict, replacements=self.replacements)


@dataclass
class RecordingMetrics:
    decisions: list[tuple[str, str, str, str]] = field(default_factory=lambda: [])
    errors: list[tuple[str, str, str]] = field(default_factory=lambda: [])
    durations: list[str] = field(default_factory=lambda: [])
    placeholder_counts: list[tuple[str, str, int]] = field(default_factory=lambda: [])

    def decision(self, stage: str, guard: str, action: str, mode: str, /) -> None:
        self.decisions.append((stage, guard, action, mode))

    def duration(self, stage: str, guard: str, seconds: float, /) -> None:
        self.durations.append(guard)

    def error(self, stage: str, guard: str, kind: str, /) -> None:
        self.errors.append((stage, guard, kind))

    def placeholders(self, direction: str, strategy: str, count: int, /) -> None:
        self.placeholder_counts.append((direction, strategy, count))


def chunk(content: str | None = None, *, finish: Any = None, role: bool = False, index: int = 0) -> ChatChunk:
    delta = Delta(role="assistant" if role else None, content=content)
    return ChatChunk(
        id="c1", created=1, model="m", choices=(ChunkChoice(index=index, delta=delta, finish_reason=finish),)
    )


class Upstream:
    """async chunk source that records whether it was closed and how far it was read"""

    def __init__(self, chunks: Sequence[ChatChunk], *, fail_after: int | None = None) -> None:
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.closed = False
        self.served = 0

    async def gen(self) -> AsyncIterator[ChatChunk]:
        try:
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i == self._fail_after:
                    raise RuntimeError("upstream broke")
                self.served += 1
                yield c
        finally:
            self.closed = True


def text_chunks(parts: Sequence[str], *, finish: Any = "stop") -> list[ChatChunk]:
    out = [chunk(p, role=i == 0) for i, p in enumerate(parts)]
    out.append(chunk(finish=finish))
    return out


def released_text(chunks: Sequence[ChatChunk]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
