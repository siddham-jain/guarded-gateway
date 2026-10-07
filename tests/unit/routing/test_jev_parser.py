import copy
from typing import Any

import pytest

from gg.routing.config import JevConfig
from gg.routing.scorers.jev.parser import JevParseError, ResponseMeta, ScoreDeriver
from tests.unit.routing.support import all_fixtures, fixture, qset

CFG = JevConfig()
DERIVER = ScoreDeriver(
    scorer_version="jev-1.13.0:qset-v1:state-v1:strong_helps",
    model=CFG.model,
    tier_options=qset().tier_options,
    strong_tiers=CFG.strong_tiers,
    price_per_mtok_input_usd=CFG.price_per_mtok_input_usd,
)
STRONG = {"frontier", "frontier_reasoning"}


def body(name: str = "p13") -> dict[str, Any]:
    return copy.deepcopy(fixture(name)["response"])


@pytest.mark.parametrize("name", all_fixtures())
def test_real_fixtures_parse_and_separate_tiers(name: str) -> None:
    recorded = fixture(name)
    score = DERIVER.derive(recorded["response"], ResponseMeta(latency_ms=300, request_id="req_1"))
    assert 0 <= score.score <= 1
    assert score.score == score.raw_score
    assert not score.fallback
    assert score.has_probabilities
    assert score.response_model == "jev-1.13.0"
    assert score.input_tokens is not None
    assert score.cost_usd == pytest.approx(score.input_tokens * 0.042 / 1e6)
    assert score.tier == recorded["response"]["answers"]["tier"]["choice"]
    expected_strong = recorded["expected_tier"] in STRONG
    assert (score.score >= 0.5) is expected_strong
    assert set(score.raw) == {"model", "answers", "usage"}


def test_score_is_strong_tier_mass_and_fields() -> None:
    score = DERIVER.derive(body("p13"), ResponseMeta())
    probs = body("p13")["answers"]["tier"]["probabilities"]
    assert score.score == pytest.approx(probs["frontier"] + probs["frontier_reasoning"])
    assert score.confidence == pytest.approx(max(probs.values()))
    assert score.task_type == body("p13")["answers"]["task_type"]["choice"]
    assert score.difficulty is not None
    assert 0 <= score.difficulty <= 1
    assert score.guard_signals["routing_claim_present"] < 0.5
    assert score.reasoning_effort_hint in {"low", "medium", "high"}


def test_probabilities_are_renormalised() -> None:
    raw = body()
    raw["answers"]["tier"]["probabilities"] = {
        "small": 0.0,
        "mid": 0.21,
        "frontier": 0.72,
        "frontier_reasoning": 0.08,
    }
    score = DERIVER.derive(raw, ResponseMeta())
    assert score.score == pytest.approx(0.80 / 1.01)
    assert sum(score.tier_probabilities.values()) == pytest.approx(1)


def test_ties_go_to_the_more_capable_option() -> None:
    raw = body()
    raw["answers"]["tier"]["probabilities"] = {
        "small": 0.0,
        "mid": 0.5,
        "frontier": 0.5,
        "frontier_reasoning": 0.0,
    }
    assert DERIVER.derive(raw, ResponseMeta()).tier == "frontier"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("answers"),
        lambda b: b["answers"].pop("tier"),
        lambda b: b["answers"]["tier"].update(type="score"),
        lambda b: b["answers"]["tier"]["probabilities"].pop("mid"),
        lambda b: b["answers"]["tier"]["probabilities"].update(extra=0.0),
        lambda b: b["answers"]["tier"]["probabilities"].update(mid="high"),
        lambda b: b["answers"]["tier"]["probabilities"].update(mid=1.5),
        lambda b: b["answers"]["tier"].update(
            probabilities={"small": 0.2, "mid": 0.2, "frontier": 0.2, "frontier_reasoning": 0.2}
        ),
    ],
)
def test_invalid_tier_answers_raise(mutate: Any) -> None:
    raw = body()
    mutate(raw)
    with pytest.raises(JevParseError):
        DERIVER.derive(raw, ResponseMeta())


def test_missing_secondary_answers_are_none_not_fallback() -> None:
    raw = body()
    for name in (
        "strong_helps",
        "difficulty",
        "task_type",
        "needs_multistep_reasoning",
        "routing_claim_present",
    ):
        raw["answers"].pop(name)
    raw.pop("usage")
    score = DERIVER.derive(raw, ResponseMeta())
    assert score.strong_helps is None
    assert score.difficulty is None
    assert score.task_type is None
    assert score.needs_reasoning is None
    assert score.guard_signals == {"routing_claim_present": 0.0}
    assert score.cost_usd is None


def test_out_of_range_secondary_answers_are_dropped() -> None:
    raw = body()
    raw["answers"]["difficulty"]["score"] = 7
    raw["answers"]["strong_helps"]["noul"] = -1
    score = DERIVER.derive(raw, ResponseMeta())
    assert score.difficulty is None
    assert score.strong_helps is None


@pytest.mark.parametrize(
    ("probs", "reasoning", "hint"),
    [
        ({"small": 0, "mid": 0, "frontier": 0.4, "frontier_reasoning": 0.6}, 0.1, "high"),
        ({"small": 0, "mid": 0, "frontier": 1.0, "frontier_reasoning": 0}, 0.85, "high"),
        ({"small": 0, "mid": 0, "frontier": 1.0, "frontier_reasoning": 0}, 0.6, "medium"),
        ({"small": 0, "mid": 0, "frontier": 1.0, "frontier_reasoning": 0}, 0.2, "low"),
        ({"small": 0, "mid": 1.0, "frontier": 0, "frontier_reasoning": 0}, 0.7, "low"),
        ({"small": 1.0, "mid": 0, "frontier": 0, "frontier_reasoning": 0}, 0.1, "none"),
    ],
)
def test_effort_hint_table(probs: dict[str, float], reasoning: float, hint: str) -> None:
    raw = body()
    raw["answers"]["tier"]["probabilities"] = probs
    raw["answers"]["needs_multistep_reasoning"]["noul"] = reasoning
    assert DERIVER.derive(raw, ResponseMeta()).reasoning_effort_hint == hint


def test_model_mismatch_is_detected() -> None:
    raw = body()
    raw["model"] = "jev-1.14.0"
    assert DERIVER.model_mismatch(DERIVER.derive(raw, ResponseMeta()))
    assert not DERIVER.model_mismatch(DERIVER.derive(body(), ResponseMeta()))


def _deriver(signal: str) -> ScoreDeriver:
    cfg = JevConfig()
    return ScoreDeriver(
        scorer_version="v",
        model=cfg.model,
        tier_options=qset().tier_options,
        strong_tiers=cfg.strong_tiers,
        price_per_mtok_input_usd=cfg.price_per_mtok_input_usd,
        score_signal=signal,
    )


def test_score_signal_picks_strong_helps_or_tier_mass() -> None:
    body = copy.deepcopy(fixture("p14")["response"])
    body["answers"]["strong_helps"] = {"type": "noul", "noul": 0.42}
    tier_mass = _deriver("tier").derive(body, ResponseMeta()).score
    assert _deriver("strong_helps").derive(body, ResponseMeta()).score == 0.42
    assert tier_mass != 0.42


def test_missing_strong_helps_falls_back_to_tier_mass() -> None:
    body = copy.deepcopy(fixture("p14")["response"])
    del body["answers"]["strong_helps"]
    assert (
        _deriver("strong_helps").derive(body, ResponseMeta()).score
        == _deriver("tier").derive(body, ResponseMeta()).score
    )
