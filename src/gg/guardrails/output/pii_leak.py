"""tier 0 output: pii the model produced that did not come from this request's own vault"""

from typing import Annotated, Literal

from pydantic import Field

from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Redaction, Streaming, finding
from gg.guardrails.pii_patterns import PiiDetector
from gg.guardrails.registry import GuardDeps
from gg.guardrails.rules import RUN_CONTINUATION


class PiiLeakCfg(StrictModel):
    entities: tuple[str, ...] = Field(
        default=("EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "IP_ADDRESS"),
        min_length=1,
    )
    action: Literal["redact", "block"] = "redact"
    ignore_vault_values: bool = True
    allow_values: tuple[str, ...] = ()
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.5


class PiiLeak:
    name: str = "pii_leak"
    stage: GuardStage = "output"
    tier: int = 0
    streaming: Streaming = "windowed"

    def __init__(self, cfg: PiiLeakCfg) -> None:
        self._detector = PiiDetector(cfg.entities, allow_values=cfg.allow_values, min_score=cfg.min_score)
        self._verdict = Verdict.BLOCK if cfg.action == "block" else Verdict.REDACT
        self._ignore_vault = cfg.ignore_vault_values

    def holdback(self, text: str, /) -> int:
        return self._detector.holdback(text)

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        spans: list[Redaction] = []
        for seg in gctx.segments:
            for m in self._detector.find(seg.text):
                if self._ignore_vault and gctx.vault.is_vault_value(m.label, seg.text[m.start : m.end]):
                    continue
                spans.append(Redaction(seg.index, m.start, m.end, m.label, m.score, RUN_CONTINUATION))
        if not spans:
            return finding(self.name, "output")
        labels = tuple(sorted({s.label for s in spans}))
        top = max(s.score for s in spans)
        return finding(
            self.name,
            "output",
            self._verdict,
            score=top,
            reason=labels[0],
            labels=labels,
            redactions=tuple(spans),
        )


def create(cfg: PiiLeakCfg, deps: GuardDeps) -> PiiLeak:
    return PiiLeak(cfg)
