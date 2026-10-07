from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from gg.routing.base import EffortHint, RoutingScore


class JevParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ResponseMeta:
    latency_ms: float = 0.0
    request_id: str | None = None
    cached: bool = False


class ScoreDeriver:
    """turns a jev response body into a RoutingScore; pure, so cached answers re-derive under new config"""

    def __init__(
        self,
        *,
        scorer_version: str,
        model: str,
        tier_options: Sequence[str],
        strong_tiers: Sequence[str],
        price_per_mtok_input_usd: float,
        score_signal: str = "tier",
    ) -> None:
        self._version = scorer_version
        self._signal = score_signal
        self._model = model
        self._options = tuple(tier_options)
        self._strong = frozenset(strong_tiers)
        self._price = price_per_mtok_input_usd

    def derive(self, body: Mapping[str, Any], meta: ResponseMeta) -> RoutingScore:
        answers = _mapping(body.get("answers"), "answers")
        probs = self._tier_probabilities(_mapping(answers.get("tier"), "answers.tier"))
        s_tier = min(1.0, sum(p for k, p in probs.items() if k in self._strong))
        strong_helps = _noul(answers, "strong_helps")
        # a missing strong_helps answer falls back to the tier score rather than routing everything weak
        score = strong_helps if self._signal == "strong_helps" and strong_helps is not None else s_tier
        needs_reasoning = _noul(answers, "needs_multistep_reasoning")
        task = _optional_mapping(answers.get("task_type"))
        usage = _optional_mapping(body.get("usage"))
        input_tokens = usage.get("input_tokens")
        tokens = input_tokens if isinstance(input_tokens, int) else None
        model = body.get("model")
        return RoutingScore(
            score=score,
            raw_score=score,
            scorer="jev",
            scorer_version=self._version,
            has_probabilities=True,
            tier=max(probs, key=lambda k: (probs[k], self._options.index(k))),
            tier_probabilities=probs,
            difficulty=_difficulty(answers),
            strong_helps=strong_helps,
            needs_reasoning=needs_reasoning,
            task_type=task.get("choice") if isinstance(task.get("choice"), str) else None,
            task_type_probabilities=_float_map(task.get("probabilities")),
            reasoning_effort_hint=self._effort_hint(s_tier, probs, needs_reasoning),
            guard_signals={"routing_claim_present": _noul(answers, "routing_claim_present") or 0.0},
            confidence=max(probs.values()),
            latency_ms=meta.latency_ms,
            input_tokens=tokens,
            cost_usd=tokens * self._price / 1e6 if tokens is not None else None,
            cached=meta.cached,
            request_id=meta.request_id,
            response_model=model if isinstance(model, str) else None,
            raw={"model": model, "answers": answers, "usage": usage},
        )

    def model_mismatch(self, score: RoutingScore) -> bool:
        return score.response_model is not None and score.response_model != self._model

    def _tier_probabilities(self, tier: Mapping[str, Any]) -> dict[str, float]:
        if tier.get("type") != "choice":
            raise JevParseError("answers.tier is not a choice answer")
        raw = _mapping(tier.get("probabilities"), "answers.tier.probabilities")
        if set(raw) != set(self._options):
            raise JevParseError("answers.tier.probabilities options differ from the question set")
        probs: dict[str, float] = {}
        for option in self._options:
            value = raw[option]
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise JevParseError(f"answers.tier.probabilities.{option} is not a probability")
            probs[option] = float(value)
        total = sum(probs.values())
        # jev rounds to two decimals, so sums drift a little from 1
        if not 0.9 <= total <= 1.1:
            raise JevParseError("answers.tier.probabilities do not sum to 1")
        return {k: v / total for k, v in probs.items()}

    @staticmethod
    def _effort_hint(s_tier: float, probs: Mapping[str, float], needs_reasoning: float | None) -> EffortHint:
        reasoning = needs_reasoning or 0.0
        if s_tier >= 0.5:
            if probs.get("frontier_reasoning", 0.0) >= 0.5 or reasoning >= 0.8:
                return "high"
            return "medium" if reasoning >= 0.5 else "low"
        return "low" if reasoning >= 0.6 else "none"


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise JevParseError(f"{where} is missing or not an object")
    return cast("dict[str, Any]", value)


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _noul(answers: Mapping[str, Any], name: str) -> float | None:
    value = _optional_mapping(answers.get(name)).get("noul")
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
        return None
    return float(value)


def _difficulty(answers: Mapping[str, Any]) -> float | None:
    value = _optional_mapping(answers.get("difficulty")).get("score")
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 4:
        return None
    return float(value) / 4


def _float_map(value: Any) -> dict[str, float]:
    return {
        str(k): float(v)
        for k, v in _optional_mapping(value).items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
