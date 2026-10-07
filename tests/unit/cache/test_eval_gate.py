from gg.cache.embedders.hashing import HashingEmbedder
from gg.cache.eval.gate import baseline_from, outcome, regressions, run_gate
from gg.cache.eval.pairs import Pair
from gg.cache.eval.sweep import Scored


def _pair(pid: str, a: str, b: str, hit: bool, split: str = "test") -> Pair:
    return Pair.model_validate(
        {"id": pid, "anchor": a, "candidate": b, "should_hit": hit, "category": "paraphrase", "split": split}
    )


PAIRS = [
    _pair("p0001", "what is the capital of france", "What is the capital of France?", True),
    _pair("p0002", "what is 17 times 23", "what is 17 times 24", False),
    _pair("p0003", "explain dns", "explain tcp handshakes in depth", False, "dev"),
]


def test_outcome_per_label() -> None:
    hit, no_hit = PAIRS[0], PAIRS[1]
    assert outcome(Scored(hit, 0.01), 0.05) == "hit"
    assert outcome(Scored(hit, 0.2), 0.05) == "missed"
    assert outcome(Scored(no_hit, 0.01), 0.05) == "false_hit"
    assert outcome(Scored(no_hit, None), 0.05) == "rejected"


def test_only_no_hit_pairs_regress() -> None:
    items = [
        {"id": "p0001", "should_hit": True, "outcome": "missed"},
        {"id": "p0002", "should_hit": False, "outcome": "false_hit"},
        {"id": "p0003", "should_hit": False, "outcome": "rejected"},
        {"id": "p0004", "should_hit": False, "outcome": "false_hit"},
    ]
    baseline = {"items": {"p0001": "hit", "p0002": "rejected", "p0003": "false_hit"}}
    assert regressions(items, baseline) == [{"id": "p0002", "from": "rejected", "to": "false_hit"}]
    assert regressions(items, None) == []


async def test_run_gate_scores_at_the_threshold_and_records_a_baseline() -> None:
    result = await run_gate(PAIRS, HashingEmbedder(64), tau=0.05, baseline={"items": {"p0002": "rejected"}})
    assert {i["id"]: i["outcome"] for i in result["items"]} == {
        "p0001": "hit",
        "p0002": "rejected",
        "p0003": "rejected",
    }
    assert result["status"] == "pass"
    assert result["metrics"]["precision"] == {"value": 1.0, "n": 1, "k": 1, "direction": "higher"}
    assert result["metrics"]["dev_precision"]["value"] is None
    baseline = baseline_from(result)
    assert baseline["items"]["p0002"] == "rejected"
    assert baseline["recorded_from"] == {"embedder": "hashing-64:64", "threshold": 0.05, "num_sig": True}
    assert baseline["metrics"] == {"precision": 1.0, "hit_rate": 1.0}


async def test_run_gate_flags_a_pair_that_starts_hitting_without_num_sig() -> None:
    baseline = {"items": {"p0002": "rejected"}}
    result = await run_gate(PAIRS, HashingEmbedder(64), tau=2.0, baseline=baseline, num_sig=False)
    assert result["regressions"] == [{"id": "p0002", "from": "rejected", "to": "false_hit"}]
    assert result["status"] == "fail"
