"""tier 1: presidio (spacy ner + recognizers) next to pii_regex; spans merge in the same vault"""

import functools
from collections.abc import Sequence
from typing import Annotated, Literal, Protocol

from pydantic import Field, field_validator

from gg.core.aio import CpuExecutor
from gg.core.guard_types import Verdict
from gg.core.schema import Role, StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Redaction, Streaming, finding
from gg.guardrails.ml.common import ML_DISABLED
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.registry import GuardDeps

type EntityAction = Literal["redact", "block", "off"]

# presidio entity -> placeholder label; labels shared with pii_regex merge into one placeholder per value
LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "NRP": "NRP",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP",
    "US_SSN": "SSN",
    "US_PASSPORT": "PASSPORT",
    "UK_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "LICENSE",
    "UK_DRIVING_LICENCE": "LICENSE",
    "US_ITIN": "ITIN",
    "US_BANK_NUMBER": "BANK",
    "UK_NHS": "NHS",
    "UK_NINO": "NINO",
    "CRYPTO": "CRYPTO",
    "MEDICAL_LICENSE": "MEDLICENSE",
    "MAC_ADDRESS": "MAC",
    "DATE_OF_BIRTH": "DOB",
    "STREET_ADDRESS": "ADDRESS",
    "UK_SORT_CODE": "SORTCODE",
}


class EntityHitLike(Protocol):
    @property
    def entity(self) -> str: ...
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def score(self) -> float: ...


class Analyzer(Protocol):
    def analyze(self, text: str, entities: Sequence[str], min_score: float, /) -> Sequence[EntityHitLike]: ...


class PiiNerCfg(StrictModel):
    entities: dict[str, EntityAction] = Field(
        default_factory=lambda: {
            "PERSON": "off",
            "LOCATION": "off",
            "PHONE_NUMBER": "redact",
            "US_PASSPORT": "redact",
            "UK_PASSPORT": "redact",
            "US_DRIVER_LICENSE": "redact",
            "US_ITIN": "redact",
            "US_BANK_NUMBER": "redact",
            "UK_NHS": "redact",
            "UK_NINO": "redact",
            "CRYPTO": "redact",
            "DATE_OF_BIRTH": "redact",
            "STREET_ADDRESS": "redact",
            "UK_SORT_CODE": "redact",
        }
    )
    allow_values: tuple[str, ...] = ()
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.5
    roles: tuple[Role, ...] = ("user", "assistant", "tool")
    # ner cost grows with length; text past this is covered by pii_regex only
    max_scan_chars: Annotated[int, Field(ge=200, le=100_000)] = 8000
    spacy_model: Literal["en_core_web_sm"] = "en_core_web_sm"

    @field_validator("entities")
    @classmethod
    def _known(cls, value: dict[str, EntityAction]) -> dict[str, EntityAction]:
        unknown = set(value) - set(LABELS)
        if unknown:
            raise ValueError(f"unsupported entities {sorted(unknown)}; supported: {sorted(LABELS)}")
        return value


class PiiNer:
    name: str = "pii_ner"
    stage: GuardStage = "input"
    tier: int = 1
    streaming: Streaming = "windowed"
    redacts_upstream: bool = True

    def __init__(self, cfg: PiiNerCfg, analyzer: Resource[Analyzer] | None, cpu: CpuExecutor | None) -> None:
        self._cfg = cfg
        self._roles = frozenset(cfg.roles)
        self._actions = {e: a for e, a in cfg.entities.items() if a != "off"}
        self._allow = frozenset(v.casefold() for v in cfg.allow_values)
        self._analyzer = analyzer
        self._cpu = cpu

    def _scan(self, analyzer: Analyzer, texts: Sequence[str]) -> list[list[EntityHitLike]]:
        entities = sorted(self._actions)
        limit = self._cfg.max_scan_chars
        return [list(analyzer.analyze(t[:limit], entities, self._cfg.min_score)) for t in texts]

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        if self._analyzer is None or self._cpu is None:
            return finding(self.name, "input", reason=ML_DISABLED)
        segments = [s for s in gctx.segments if s.role in self._roles and s.text.strip()]
        if not segments or not self._actions:
            return finding(self.name, "input")
        analyzer = await self._analyzer.get()
        hits = await self._cpu.run(functools.partial(self._scan, analyzer, [s.text for s in segments]))
        spans: list[Redaction] = []
        blocked: set[str] = set()
        found: set[str] = set()
        top = 0.0
        for seg, seg_hits in zip(segments, hits, strict=True):
            for hit in seg_hits:
                if seg.text[hit.start : hit.end].strip().casefold() in self._allow:
                    continue
                label = LABELS[hit.entity]
                found.add(label)
                top = max(top, hit.score)
                if self._actions[hit.entity] == "block":
                    blocked.add(label)
                else:
                    spans.append(Redaction(seg.index, hit.start, hit.end, label, hit.score))
        if not found:
            return finding(self.name, "input")
        labels = tuple(sorted(found))
        return finding(
            self.name,
            "input",
            Verdict.BLOCK if blocked else Verdict.REDACT,
            score=top,
            reason=sorted(blocked)[0] if blocked else labels[0],
            labels=labels,
            redactions=tuple(spans),
        )


def create(cfg: PiiNerCfg, deps: GuardDeps) -> PiiNer:
    runtime = deps.models
    if runtime is None:
        return PiiNer(cfg, None, None)

    def load() -> Analyzer:
        from gg.guardrails.ml.presidio import PresidioAnalyzer

        return PresidioAnalyzer(cfg.spacy_model)

    async def run() -> Analyzer:
        return await runtime.cpu.run(load)

    return PiiNer(cfg, runtime.resource(f"presidio:{cfg.spacy_model}", run), runtime.cpu)
