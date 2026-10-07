"""post-hoc output check: each reply sentence scored against the request's context with hhem; flag only"""

import functools
import re
from typing import Annotated

from pydantic import Field, field_validator

from gg.core.aio import CpuExecutor
from gg.core.guard_types import Verdict
from gg.core.schema import ChatRequest, Role, StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.ml.artefacts import model_spec
from gg.guardrails.ml.common import ML_DISABLED, PairScorer
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.registry import GuardDeps

NO_CONTEXT = "no_context"
_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class GroundingCfg(StrictModel):
    model: str = "hhem-2.1-open"
    # where the grounding documents live; user turns are questions, not evidence
    context_roles: tuple[Role, ...] = ("system", "developer", "tool")
    min_context_chars: Annotated[int, Field(ge=1)] = 200
    # ~1.5k tokens; the premise is repeated for every sentence, so this bounds the cost
    max_context_chars: Annotated[int, Field(ge=200, le=24_000)] = 6000
    max_sentences: Annotated[int, Field(ge=1, le=64)] = 12
    min_sentence_chars: Annotated[int, Field(ge=1)] = 20
    flag_below: Annotated[float, Field(gt=0, le=1)] = 0.5
    lazy: bool = True

    @field_validator("model")
    @classmethod
    def _known(cls, value: str) -> str:
        model_spec(value)
        return value


def context_of(request: ChatRequest, roles: frozenset[Role]) -> str:
    return "\n\n".join(m.text() for m in request.messages if m.role in roles and m.text().strip())


def sentences(text: str, *, min_chars: int, limit: int) -> list[str]:
    prose = _FENCE_RE.sub(" ", text)
    out = [s.strip() for s in _SENTENCE_RE.split(prose)]
    return [s for s in out if len(s) >= min_chars][:limit]


class Grounding:
    name: str = "grounding"
    stage: GuardStage = "output"
    tier: int = 3
    streaming: Streaming = "post_hoc"

    def __init__(
        self, cfg: GroundingCfg, model: Resource[PairScorer] | None, cpu: CpuExecutor | None
    ) -> None:
        self._cfg = cfg
        self._roles = frozenset(cfg.context_roles)
        self._model = model
        self._cpu = cpu

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        if self._model is None or self._cpu is None:
            return finding(self.name, "output", reason=ML_DISABLED)
        cfg = self._cfg
        premise = context_of(gctx.request, self._roles)
        if len(premise) < cfg.min_context_chars:
            return finding(self.name, "output", reason=NO_CONTEXT)
        claims = [
            s
            for seg in gctx.segments
            for s in sentences(seg.text, min_chars=cfg.min_sentence_chars, limit=cfg.max_sentences)
        ][: cfg.max_sentences]
        if not claims:
            return finding(self.name, "output")
        model = await self._model.get()
        scores = await self._cpu.run(
            functools.partial(model.scores, premise[: cfg.max_context_chars], claims)
        )
        low = min(scores)
        verdict = Verdict.FLAG if low < cfg.flag_below else Verdict.ALLOW
        return finding(
            self.name, "output", verdict, score=low, reason="ungrounded" if verdict is Verdict.FLAG else ""
        )


def create(cfg: GroundingCfg, deps: GuardDeps) -> Grounding:
    runtime = deps.models
    if runtime is None:
        return Grounding(cfg, None, None)
    return Grounding(cfg, runtime.pair_scorer(cfg.model, lazy=cfg.lazy), runtime.cpu)
