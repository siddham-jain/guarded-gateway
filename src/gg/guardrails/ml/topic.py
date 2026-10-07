"""tier 2: latest user turn vs hand-written topic exemplars, embedded with the semantic cache's embedder"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from gg.cache.base import Embedder
from gg.core.guard_types import Verdict
from gg.core.jsonutil import canonical_json
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Segment, Streaming, finding
from gg.guardrails.ml.common import ML_DISABLED
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.registry import GuardDeps

OFF_TOPIC = "off_topic"


class TopicSpec(StrictModel):
    threshold: Annotated[float, Field(gt=0, le=1)]
    exemplars: tuple[str, ...] = Field(min_length=1)


class TopicCfg(StrictModel):
    # block (or flag) when the turn is at least `threshold` similar to any exemplar of a deny topic
    deny: dict[str, TopicSpec] = Field(default_factory=lambda: {})
    # allow-list mode: block when the turn is below threshold for every allow topic
    allow: dict[str, TopicSpec] | None = None
    action: Literal["block", "flag"] = "block"
    max_chars: Annotated[int, Field(ge=16, le=8000)] = 2000

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.deny and not self.allow:
            raise ValueError("topic needs at least one deny or allow topic")
        return self


type Vector = Sequence[float]


@dataclass(frozen=True, slots=True)
class TopicIndex:
    """exemplar vectors per topic; vectors are unit length, so a dot product is the cosine"""

    deny: Mapping[str, tuple[float, tuple[Vector, ...]]]
    allow: Mapping[str, tuple[float, tuple[Vector, ...]]]

    @staticmethod
    def best(
        vector: Vector, topics: Mapping[str, tuple[float, tuple[Vector, ...]]]
    ) -> list[tuple[str, float, float]]:
        """(topic, similarity, threshold) per topic, most similar first"""
        out = [
            (name, max(sum(a * b for a, b in zip(vector, ex, strict=True)) for ex in exemplars), threshold)
            for name, (threshold, exemplars) in topics.items()
        ]
        return sorted(out, key=lambda t: -t[1])


async def build_index(cfg: TopicCfg, embedder: Embedder) -> TopicIndex:
    allow = cfg.allow or {}
    texts = [e for spec in (*cfg.deny.values(), *allow.values()) for e in spec.exemplars]
    vectors = iter(await embedder.embed(texts))

    def take(topics: Mapping[str, TopicSpec]) -> dict[str, tuple[float, tuple[Vector, ...]]]:
        return {name: (s.threshold, tuple(next(vectors) for _ in s.exemplars)) for name, s in topics.items()}

    return TopicIndex(deny=take(cfg.deny), allow=take(allow))


def latest_user_text(segments: Sequence[Segment]) -> str:
    users = [s for s in segments if s.role == "user" and s.kind == "content"]
    if not users:
        return ""
    last = users[-1].msg
    return "\n".join(s.view for s in users if s.msg == last)


class Topic:
    name: str = "topic"
    stage: GuardStage = "input"
    tier: int = 2
    streaming: Streaming = "windowed"

    def __init__(self, cfg: TopicCfg, index: Resource[TopicIndex] | None, embedder: Embedder | None) -> None:
        self._cfg = cfg
        self._index = index
        self._embedder = embedder
        self._hit = Verdict.BLOCK if cfg.action == "block" else Verdict.FLAG

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        if self._index is None or self._embedder is None:
            return finding(self.name, "input", reason=ML_DISABLED)
        text = latest_user_text(gctx.segments)[: self._cfg.max_chars]
        if not text.strip():
            return finding(self.name, "input")
        index = await self._index.get()
        [vector] = await self._embedder.embed([text])
        denied = [t for t in TopicIndex.best(vector, index.deny) if t[1] >= t[2]]
        if denied:
            topic, sim, _ = denied[0]
            return finding(self.name, "input", self._hit, score=sim, reason=topic, labels=(topic,))
        if index.allow:
            allowed = TopicIndex.best(vector, index.allow)
            if not any(sim >= threshold for _, sim, threshold in allowed):
                top = allowed[0][1]
                return finding(
                    self.name, "input", self._hit, score=top, reason=OFF_TOPIC, labels=(OFF_TOPIC,)
                )
        return finding(self.name, "input")


def create(cfg: TopicCfg, deps: GuardDeps) -> Topic:
    runtime = deps.models
    if runtime is None:
        return Topic(cfg, None, None)
    embedder = runtime.embedder
    if embedder is None:
        raise ValueError("the topic guard needs the shared embedder (pass embedder= to build_guardrails)")
    digest = hashlib.sha256(
        canonical_json(cfg.model_dump(mode="json", include={"deny", "allow"}))
    ).hexdigest()

    async def load() -> TopicIndex:
        return await build_index(cfg, embedder)

    index = runtime.resource(f"topic:{embedder.name}:{digest[:12]}", load)
    return Topic(cfg, index, embedder)
