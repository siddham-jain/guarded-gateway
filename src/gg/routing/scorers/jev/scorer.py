from collections.abc import Mapping
from typing import Any

import structlog

from gg.core.jsonutil import dumps
from gg.routing.base import RoutingRequest, RoutingScore
from gg.routing.config import JevConfig
from gg.routing.scorers.jev.client import JevClient, JevFailure
from gg.routing.scorers.jev.parser import JevParseError, ResponseMeta, ScoreDeriver
from gg.routing.scorers.jev.questions import QuestionSet
from gg.routing.scorers.jev.state import StateBuilder

log = structlog.get_logger("gg.routing.jev")


class JevScorer:
    """scores a prompt with one batched jev call; implements CacheableScorer"""

    name = "jev"

    def __init__(
        self,
        client: JevClient,
        builder: StateBuilder,
        qset: QuestionSet,
        cfg: JevConfig,
        *,
        deadline_s: float,
    ) -> None:
        self._client = client
        self._builder = builder
        self._qset = qset
        self._cfg = cfg
        self._deadline_s = deadline_s
        # thresholds are only valid for one score signal, so it is part of the version
        self._version = f"{cfg.model}:{qset.version}:{builder.version}:{cfg.score_signal}"
        self._deriver = ScoreDeriver(
            scorer_version=self._version,
            model=cfg.model,
            tier_options=qset.tier_options,
            strong_tiers=cfg.strong_tiers,
            price_per_mtok_input_usd=cfg.price_per_mtok_input_usd,
            score_signal=cfg.score_signal,
        )

    @property
    def version(self) -> str:
        return self._version

    def cache_key(self, req: RoutingRequest, /) -> str | None:
        state = self._builder.build(req)
        return None if state.unscorable else state.cache_key(self._version)

    def from_raw(self, raw: Mapping[str, Any], req: RoutingRequest, /) -> RoutingScore:
        return self._deriver.derive(raw, ResponseMeta(cached=True))

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        state = self._builder.build(req)
        if state.state is None:
            return RoutingScore.fallback_for(self.name, self._version, "unscorable")
        body = dumps({"model": self._cfg.model, "state": state.state, "questions": self._qset.questions})
        result = await self._client.ask(body, self._deadline_s)
        if isinstance(result, JevFailure):
            return RoutingScore.fallback_for(
                self.name, self._version, result.reason, result.latency_ms, request_id=result.request_id
            )
        meta = ResponseMeta(latency_ms=result.latency_ms, request_id=result.request_id)
        try:
            score = self._deriver.derive(result.body, meta)
        except JevParseError as exc:
            log.warning("jev.parse_error", error=str(exc), request_id=result.request_id)
            return RoutingScore.fallback_for(
                self.name, self._version, "parse_error", result.latency_ms, request_id=result.request_id
            )
        if self._deriver.model_mismatch(score):
            log.warning("jev.model_mismatch", pinned=self._cfg.model, answered=score.response_model)
        return score
