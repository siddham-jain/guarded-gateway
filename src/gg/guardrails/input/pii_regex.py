"""tier 1: regex pii with checksums; reversible redaction through the vault, per-entity actions"""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from gg.core.guard_types import Verdict
from gg.core.schema import Role, StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Redaction, Streaming, finding
from gg.guardrails.pii_patterns import ENTITIES, PiiDetector
from gg.guardrails.registry import GuardDeps

type EntityAction = Literal["redact", "block", "off"]


class PiiRegexCfg(StrictModel):
    entities: dict[str, EntityAction] = Field(
        default_factory=lambda: {
            "EMAIL_ADDRESS": "redact",
            "PHONE_NUMBER": "redact",
            "CREDIT_CARD": "redact",
            "IBAN_CODE": "redact",
            "IP_ADDRESS": "redact",
            "US_SSN": "block",
        }
    )
    allow_values: tuple[str, ...] = ()
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.5
    roles: tuple[Role, ...] = ("user", "assistant", "tool")

    @field_validator("entities")
    @classmethod
    def _known(cls, value: dict[str, EntityAction]) -> dict[str, EntityAction]:
        unknown = set(value) - ENTITIES
        if unknown:
            raise ValueError(f"unsupported entities {sorted(unknown)}; supported: {sorted(ENTITIES)}")
        return value


class PiiRegex:
    name: str = "pii_regex"
    stage: GuardStage = "input"
    tier: int = 1
    streaming: Streaming = "windowed"
    redacts_upstream: bool = True

    def __init__(self, cfg: PiiRegexCfg) -> None:
        self._roles = frozenset(cfg.roles)
        self._actions = {e: a for e, a in cfg.entities.items() if a != "off"}
        self._detector = PiiDetector(self._actions, allow_values=cfg.allow_values, min_score=cfg.min_score)

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        spans: list[Redaction] = []
        blocked: set[str] = set()
        found: set[str] = set()
        top = 0.0
        for seg in gctx.segments:
            if seg.role not in self._roles:
                continue
            for m in self._detector.find(seg.text):
                found.add(m.label)
                top = max(top, m.score)
                if self._actions[m.entity] == "block":
                    blocked.add(m.label)
                else:
                    spans.append(Redaction(seg.index, m.start, m.end, m.label, m.score))
        if not found:
            return finding(self.name, "input")
        verdict = Verdict.BLOCK if blocked else Verdict.REDACT
        labels = tuple(sorted(found))
        reason = sorted(blocked)[0] if blocked else labels[0]
        return finding(
            self.name, "input", verdict, score=top, reason=reason, labels=labels, redactions=tuple(spans)
        )


def create(cfg: PiiRegexCfg, deps: GuardDeps) -> PiiRegex:
    return PiiRegex(cfg)
