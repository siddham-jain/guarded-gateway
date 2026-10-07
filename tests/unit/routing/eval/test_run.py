from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from gg.routing.base import RoutingRequest, RoutingScore
from gg.routing.eval.budget import BudgetExceeded, CallEstimate, Estimate, Price, SpendCap
from gg.routing.eval.dryrun import mock_judge_params, mock_params
from gg.routing.eval.items import PairConfig, limit_items, load_items, load_suite
from gg.routing.eval.judge import JudgePrompt
from gg.routing.eval.run import Generation, RoutingEvalRun, Stores
from gg.routing.eval.scorers import FakeScorer
from gg.routing.eval.store import JsonlStore, record_key

ROOT = Path(__file__).resolve().parents[4]
SUITE = load_suite(ROOT / "evals/routing/suite.yaml")
ITEMS = limit_items(load_items(SUITE.items), 26)
PAIR = PairConfig.model_validate(
    {
        "profile": "test",
        "weak": {"model": "w"},
        "strong": {"model": "s", "extra_tokens": 100},
        "judge": {"model": "j", "max_tokens": 200},
    }
)


class FakeGenerator:
    """answers from the dry-run mock params, like the mock provider would"""

    def __init__(self, served: str | None = None) -> None:
        self.calls: list[str] = []
        self._served = served

    async def generate(
        self, model: str, messages: Sequence[Mapping[str, str]], params: Mapping[str, Any]
    ) -> Generation:
        self.calls.append(model)
        text = params["mock"]["text"]
        return Generation(
            text=text,
            finish_reason="stop",
            served_model=self._served or model,
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=0,
            cost_usd=0.001,
            billed_usd=0.001,
            latency_ms=1.0,
        )


class FailingScorer:
    name = "failing"
    version = "failing-v1"

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        return RoutingScore.fallback_for(self.name, self.version, "timeout")


def make_run(tmp_path: Path, generator: FakeGenerator, *, max_usd: float = 10.0, **kw: Any) -> RoutingEvalRun:
    prices = {"w": Price(0.1, 0.5), "s": Price(2, 10), "j": Price(1, 5)}
    return RoutingEvalRun(
        ITEMS,
        PAIR,
        stores=Stores.open(tmp_path),
        generator=generator,
        scorer=kw.pop("scorer", FakeScorer({f"eval-{i.id}": i.expected_tier for i in ITEMS})),
        prices=prices,
        judge_prompt=JudgePrompt.load(SUITE.judge_prompt),
        cap=SpendCap(max_usd),
        extra_params=mock_params,
        judge_extra=mock_judge_params,
        **kw,
    )


PHASES = ("score", "generate", "judge")


async def test_full_run_then_resume_makes_no_calls(tmp_path: Path) -> None:
    first = FakeGenerator()
    run = make_run(tmp_path, first)
    await run.run(PHASES)
    judged = sum(i.needs_judge for i in ITEMS)
    assert len(first.calls) == 2 * len(ITEMS) + 2 * judged
    collected = run.collect()
    assert len(collected.outcomes) == len(ITEMS)
    assert collected.judge_pairs == judged
    assert not any(collected.missing.values())

    second = FakeGenerator()
    rerun = make_run(tmp_path, second)
    assert rerun.estimate(PHASES).calls == 0
    await rerun.run(PHASES)
    assert second.calls == []
    assert rerun.stats["generate"].cached == 2 * len(ITEMS)
    assert rerun.collect().outcomes == collected.outcomes


async def test_killed_run_resumes_only_the_missing_calls(tmp_path: Path) -> None:
    await make_run(tmp_path, FakeGenerator()).run(PHASES)
    path = tmp_path / "generations.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    # drop the last three records and tear the one before them mid-write
    path.write_bytes(b"".join(lines[:-4]) + lines[-4][:20])
    generator = FakeGenerator()
    resumed = make_run(tmp_path, generator)
    await resumed.run(PHASES)
    assert len([c for c in generator.calls if c in ("w", "s")]) == 4


async def test_changed_params_change_the_cache_key(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGenerator())
    item = ITEMS[0]
    key = run.gen_key(item, "strong")
    other = PAIR.model_copy(
        update={"strong": PAIR.strong.model_copy(update={"params": {"reasoning_effort": "high"}})}
    )
    changed = make_run(tmp_path, FakeGenerator())
    changed.pair = other
    assert changed.gen_key(item, "strong") != key


async def test_fallback_answers_are_not_stored(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGenerator(served="other"))
    await run.run(("generate",))
    assert len(run.stores.generations) == 0
    assert run.stats["generate"].failed == 2 * len(ITEMS)


async def test_scorer_fallbacks_are_not_cached(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGenerator(), scorer=FailingScorer())
    await run.run(("score",))
    assert len(run.stores.scores) == 0
    assert run.stats["score"].failed == len(ITEMS)


async def test_spend_cap_stops_the_run_cleanly(tmp_path: Path) -> None:
    run = make_run(tmp_path, FakeGenerator(), max_usd=0.05, concurrency=1)
    estimate = run.estimate(("generate",))
    assert estimate.billed_worst_usd > 0.05
    await run.run(("generate",))
    assert run.budget_stop is not None
    assert "spend cap" in run.budget_stop
    assert 0 < len(run.stores.generations) < 2 * len(ITEMS)
    assert run.cap.spent_usd <= 0.05


def test_estimate_counts_free_tiers_at_list_price_but_not_as_billed() -> None:
    free = CallEstimate("f", 1000, 1000, Price(1, 1, billed=False))
    paid = CallEstimate("p", 1000, 1000, Price(1, 1))
    estimate = Estimate.of([free, paid], extra_billed_usd=0.5)
    assert estimate.worst_usd == pytest.approx(0.004 + 0.5)
    assert estimate.billed_worst_usd == pytest.approx(0.002 + 0.5)
    assert estimate.billed_expected_usd == pytest.approx(0.0015 + 0.5)


def test_spend_cap_reserves_worst_case_then_settles() -> None:
    cap = SpendCap(1.0)
    held = cap.reserve(0.6)
    with pytest.raises(BudgetExceeded):
        cap.reserve(0.5)
    cap.settle(held, 0.1)
    assert cap.spent_usd == pytest.approx(0.1)
    cap.reserve(0.9)


def test_store_round_trip_and_last_write_wins(tmp_path: Path) -> None:
    store = JsonlStore(tmp_path / "s.jsonl")
    key = record_key("x", {"b": 1, "a": 2})
    assert key == record_key("x", {"a": 2, "b": 1})
    store.put(key, {"v": 1})
    store.put(key, {"v": 2})
    reopened = JsonlStore(tmp_path / "s.jsonl")
    assert len(reopened) == 1
    assert reopened.get(key) == {"key": key, "v": 2}


def test_store_rejects_corruption_before_the_last_line(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text('{"key": "a"}\nnot json\n{"key": "b"}\n')
    with pytest.raises(ValueError, match="corrupt"):
        JsonlStore(path)
