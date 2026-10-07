"""dry run fakes: per-request mock scenarios for the in-process mock provider, so the whole pipeline (gateway,
stores, graders, judge parsing, sweep, report) runs with a real weak/strong gap and no network"""

from collections.abc import Mapping
from typing import Any

from gg.core.jsonutil import dumps_str
from gg.routing.eval.items import EvalItem
from gg.routing.eval.run import Role
from gg.routing.eval.scorers import unit_hash

GOOD_MARK = "[quality:good]"
POOR_MARK = "[quality:poor]"
# chance a fake answer is good, by role and the item's tier prior
_GOOD_RATE = {
    ("strong", "strong"): 0.92,
    ("strong", "weak"): 0.95,
    ("weak", "weak"): 0.9,
    ("weak", "strong"): 0.3,
}


def fake_answer(item: EvalItem, role: Role) -> str:
    good = unit_hash("fake-answer", item.id, role) < _GOOD_RATE[(role, item.expected_tier)]
    if item.needs_judge:
        body = item.reference or "A complete, specific answer."
        return f"{body} {GOOD_MARK}" if good else f"Not sure, roughly: it depends. {POOR_MARK}"
    if good and item.reference is not None:
        return item.reference
    return "I am not sure.\nAnswer: unknown"


def mock_params(item: EvalItem, role: Role) -> Mapping[str, Any]:
    return {"mock": {"text": fake_answer(item, role)}}


def mock_judge_params(item: EvalItem, answer_a: str, answer_b: str) -> Mapping[str, Any]:
    a_good, b_good = GOOD_MARK in answer_a, GOOD_MARK in answer_b
    verdict = "A" if a_good and not b_good else "B" if b_good and not a_good else "tie"
    return {"mock": {"text": dumps_str({"verdict": verdict, "reason": "fake judge for the dry run"})}}
