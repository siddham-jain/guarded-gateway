"""tier 1: gitleaks-style secret rules plus keyword-gated entropy; redacts (reversibly) or blocks"""

from typing import Literal

from gg.core.guard_types import Verdict
from gg.core.schema import Role, StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Redaction, Streaming, finding
from gg.guardrails.registry import GuardDeps
from gg.guardrails.secrets_rules import EntropyCfg, SecretDetector


class SecretsCfg(StrictModel):
    pack: str
    action: Literal["redact", "block"] = "redact"
    entropy: EntropyCfg = EntropyCfg()
    roles: tuple[Role, ...] = ("user", "assistant", "tool")


class Secrets:
    name: str = "secrets"
    stage: GuardStage = "input"
    tier: int = 1
    streaming: Streaming = "windowed"
    redacts_upstream: bool = True

    def __init__(self, cfg: SecretsCfg, detector: SecretDetector) -> None:
        self._roles = frozenset(cfg.roles)
        self._verdict = Verdict.BLOCK if cfg.action == "block" else Verdict.REDACT
        self._detector = detector

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        spans: list[Redaction] = []
        rules: set[str] = set()
        for seg in gctx.segments:
            if seg.role not in self._roles:
                continue
            for m in self._detector.find(seg.text):
                spans.append(Redaction(seg.index, m.start, m.end, "SECRET"))
                rules.add(m.rule)
            # a secret inside an encoded blob takes the whole blob with it
            for blob in seg.decoded:
                if blob.end > blob.start and self._detector.find(blob.text):
                    spans.append(Redaction(seg.index, blob.start, blob.end, "SECRET"))
                    rules.add("encoded")
        if not spans:
            return finding(self.name, "input")
        labels = tuple(sorted(rules))
        return finding(
            self.name,
            "input",
            self._verdict,
            score=1.0,
            reason=labels[0],
            labels=labels,
            redactions=tuple(spans),
        )


def create(cfg: SecretsCfg, deps: GuardDeps) -> Secrets:
    return Secrets(cfg, SecretDetector(deps.packs.secrets(cfg.pack), cfg.entropy))
