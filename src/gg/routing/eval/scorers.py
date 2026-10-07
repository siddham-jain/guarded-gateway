"""item scorers for the eval: the production jev scorer with eval-friendly timeouts, a length/keyword
heuristic baseline and the deterministic fake the dry run uses"""

import math
import re
from collections.abc import Mapping
from pathlib import Path

import httpx2

from gg.config.loader import load_file
from gg.core.clock import Clock
from gg.core.jsonutil import sha256_hex
from gg.routing.base import RoutingRequest, RoutingScore
from gg.routing.config import RoutingConfig
from gg.routing.eval.items import EvalItem, Tier
from gg.routing.scorers.jev.client import JevClient
from gg.routing.scorers.jev.questions import load_question_set
from gg.routing.scorers.jev.scorer import JevScorer
from gg.routing.scorers.jev.state import StateBuilder

# the eval is offline, so it trades the production 600 ms deadline for fewer fallbacks
EVAL_ATTEMPT_TIMEOUT_MS = 8000
EVAL_DEADLINE_S = 20.0
JEV_STATE_TOKENS_ESTIMATE = 2000


def build_jev_scorer(config_dir: Path, api_key: str, http: httpx2.AsyncClient, clock: Clock) -> JevScorer:
    cfg = load_file(config_dir / "routing.yaml", RoutingConfig)
    jev = cfg.scorer.jev.model_copy(
        update={"attempt_timeout_ms": EVAL_ATTEMPT_TIMEOUT_MS, "connect_timeout_ms": 3000}
    )
    qset = load_question_set(config_dir, jev.question_set, jev.strong_tiers)
    client = JevClient(http, api_key, jev, clock)
    return JevScorer(client, StateBuilder(jev.state), qset, jev, deadline_s=EVAL_DEADLINE_S)


def jev_cost_estimate_usd(config_dir: Path) -> float:
    cfg = load_file(config_dir / "routing.yaml", RoutingConfig)
    return JEV_STATE_TOKENS_ESTIMATE * cfg.scorer.jev.price_per_mtok_input_usd / 1_000_000


def unit_hash(*parts: str) -> float:
    """deterministic uniform [0, 1) from strings"""
    return int(sha256_hex("\x1f".join(parts))[:12], 16) / float(16**12)


class FakeScorer:
    """dry run only: the author's tier prior blurred by a hash, rounded like jev's 2-decimal probabilities"""

    name = "fake"
    version = "fake-v1"

    def __init__(self, priors: Mapping[str, Tier]) -> None:
        self._priors = priors

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        prior = 1.0 if self._priors.get(req.request_id) == "strong" else 0.0
        value = round(0.55 * prior + 0.45 * unit_hash("fake-score", req.request_id), 2)
        return RoutingScore(
            score=value, raw_score=value, scorer=self.name, scorer_version=self.version, cost_usd=0.0
        )


_CODE_RE = re.compile(r"```|\bdef |\bclass |\breturn\b|=>|;\s*$|\bSELECT\b|\bfunc\b", re.M)
_MATH_RE = re.compile(r"\d\s*[\^*/=<>]\s*\d|\^|√|∑|\bprobability\b|\bexpected\b|\bprime\b|!\s", re.I)
_REASONING_RE = re.compile(
    r"\b(prove|debug|optimi[sz]e|step by step|trade-?offs?|why|compare|explain|how many|leak|"
    r"concurren\w*|race|deadlock|architecture|design)\b",
    re.I,
)

_WEIGHTS = {"bias": -2.2, "log_chars": 0.35, "code": 0.9, "math": 0.8, "reasoning": 0.6, "turns": 0.25}


def heuristic_score(item: EvalItem) -> float:
    """hand-set logistic over length, code, maths, reasoning keywords and turn count; not fitted"""
    text = "\n".join(m.content for m in item.messages)
    user_turns = sum(1 for m in item.messages if m.role == "user")
    z = (
        _WEIGHTS["bias"]
        + _WEIGHTS["log_chars"] * math.log1p(len(text))
        + _WEIGHTS["code"] * bool(_CODE_RE.search(text))
        + _WEIGHTS["math"] * bool(_MATH_RE.search(text))
        + _WEIGHTS["reasoning"] * min(2, len(_REASONING_RE.findall(text)))
        + _WEIGHTS["turns"] * (user_turns - 1)
    )
    return round(1 / (1 + math.exp(-z)), 4)
