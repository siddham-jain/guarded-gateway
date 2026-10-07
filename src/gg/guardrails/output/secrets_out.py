"""tier 0 output: secrets in model output are replaced with a non-restorable marker"""

from typing import Literal

from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Redaction, Streaming, finding
from gg.guardrails.registry import GuardDeps
from gg.guardrails.secrets_rules import EntropyCfg, SecretDetector


class SecretsOutCfg(StrictModel):
    pack: str
    action: Literal["redact", "block"] = "redact"
    entropy: EntropyCfg = EntropyCfg()


class SecretsOut:
    name: str = "secrets_out"
    stage: GuardStage = "output"
    tier: int = 0
    streaming: Streaming = "windowed"

    def __init__(self, cfg: SecretsOutCfg, detector: SecretDetector) -> None:
        self._verdict = Verdict.BLOCK if cfg.action == "block" else Verdict.REDACT
        self._detector = detector

    def holdback(self, text: str, /) -> int:
        return self._detector.holdback(text)

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        spans: list[Redaction] = []
        rules: set[str] = set()
        for seg in gctx.segments:
            for m in self._detector.find(seg.text):
                spans.append(Redaction(seg.index, m.start, m.end, "SECRET", continuation=m.continuation))
                rules.add(m.rule)
        if not spans:
            return finding(self.name, "output")
        labels = tuple(sorted(rules))
        return finding(
            self.name,
            "output",
            self._verdict,
            score=1.0,
            reason=labels[0],
            labels=labels,
            redactions=tuple(spans),
        )


def create(cfg: SecretsOutCfg, deps: GuardDeps) -> SecretsOut:
    return SecretsOut(cfg, SecretDetector(deps.packs.secrets(cfg.pack), cfg.entropy))
