"""output tier 1: hate/abuse/profanity classifier on each stream window (or the whole reply)"""

import functools
from typing import Annotated, Self

from pydantic import Field, field_validator, model_validator

from gg.core.aio import CpuExecutor
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.ml.artefacts import model_spec
from gg.guardrails.ml.common import ML_DISABLED, TextClassifier, check_thresholds, graded
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.registry import GuardDeps


class ToxicityCfg(StrictModel):
    model: str = "granite-guardian-hap-38m"
    block_at: Annotated[float, Field(gt=0, le=1)] = 0.95
    flag_at: Annotated[float, Field(gt=0, le=1)] = 0.7
    max_windows: Annotated[int, Field(ge=1, le=64)] = 16

    @field_validator("model")
    @classmethod
    def _known(cls, value: str) -> str:
        model_spec(value)
        return value

    @model_validator(mode="after")
    def _thresholds(self) -> Self:
        check_thresholds(self.block_at, self.flag_at)
        return self


class Toxicity:
    name: str = "toxicity"
    stage: GuardStage = "output"
    tier: int = 1
    streaming: Streaming = "windowed"

    def __init__(
        self, cfg: ToxicityCfg, model: Resource[TextClassifier] | None, cpu: CpuExecutor | None
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._cpu = cpu

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        if self._model is None or self._cpu is None:
            return finding(self.name, "output", reason=ML_DISABLED)
        texts = [s.text for s in gctx.segments if s.text.strip()]
        if not texts:
            return finding(self.name, "output")
        model = await self._model.get()
        scores = await self._cpu.run(
            functools.partial(model.scores, texts, max_windows=self._cfg.max_windows)
        )
        top = max(scores)
        verdict = graded(top, block_at=self._cfg.block_at, flag_at=self._cfg.flag_at)
        return finding(self.name, "output", verdict, score=top, reason=self._cfg.model)


def create(cfg: ToxicityCfg, deps: GuardDeps) -> Toxicity:
    runtime = deps.models
    if runtime is None:
        return Toxicity(cfg, None, None)
    return Toxicity(cfg, runtime.classifier(cfg.model), runtime.cpu)
