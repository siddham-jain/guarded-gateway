from typing import Any

import pytest
from pydantic import ValidationError

from gg.evalgate.gates import added_entries, evaluate, parse_accepted, parse_suite
from gg.evalgate.report import render

SUITE = """
suite: guardrails
version: 1.0.0
gates:
  item_regressions: {mode: gate}
  input_catch_rate: {min: 0.85}
  input_fpr: {max: 0.05}
  catch_rate: {max_drop: 0.05, mode: report}
"""

ACCEPTED = """
- item: in-ben-001
  from: allowed
  to: false_positive
  reason: the new rule is right to block this
  pr: 57
"""


def _result(**metrics: float | None) -> dict[str, Any]:
    values = {"input_catch_rate": 0.9, "input_fpr": 0.0, "catch_rate": 0.9, **metrics}
    return {
        "suite": "guardrails",
        "metrics": {k: {"value": v, "n": 10, "k": 9} for k, v in values.items()},
        "items": [{"id": "in-ben-001", "category": "benign"}],
        "regressions": [],
        "gates": [],
        "status": "pass",
    }


def _gates(result: dict[str, Any]) -> dict[str, str]:
    return {g["name"]: g["status"] for g in result["gates"]}


REGRESSION = {"id": "in-ben-001", "from": "allowed", "to": "false_positive"}


def test_clean_run_passes_every_gate() -> None:
    out = evaluate(_result(), parse_suite(SUITE), {"metrics": {"catch_rate": 0.92}}, [])
    assert out["status"] == "pass"
    assert set(_gates(out).values()) == {"pass"}


def test_item_regression_fails_unless_accepted() -> None:
    spec = parse_suite(SUITE)
    result = {**_result(), "regressions": [REGRESSION]}
    failed = evaluate(result, spec, None, [])
    assert failed["status"] == "fail"
    assert _gates(failed)["item_regressions"] == "fail"
    assert failed["regressions"] == [REGRESSION]

    passed = evaluate(result, spec, None, parse_accepted(ACCEPTED))
    assert passed["status"] == "pass"
    assert passed["regressions"] == []
    assert passed["accepted_regressions"] == [REGRESSION]
    assert "accepted: in-ben-001" in passed["gates"][0]["detail"]


def test_accepted_entry_must_name_the_observed_outcome() -> None:
    wrong_to = parse_accepted(ACCEPTED.replace("to: false_positive", "to: allowed"))
    result = {**_result(), "regressions": [REGRESSION]}
    assert evaluate(result, parse_suite(SUITE), None, wrong_to)["status"] == "fail"


def test_only_entries_added_since_the_base_count() -> None:
    head = parse_accepted(ACCEPTED + ACCEPTED.replace("in-ben-001", "in-ben-002"))
    base = parse_accepted(ACCEPTED)
    assert [a.item for a in added_entries(head, base)] == ["in-ben-002"]
    assert added_entries(head, []) == head
    assert parse_accepted(None) == []
    assert parse_accepted("[]\n") == []


@pytest.mark.parametrize(
    ("metrics", "failing"),
    [
        ({"input_catch_rate": 0.8}, "input_catch_rate"),
        ({"input_fpr": 0.1}, "input_fpr"),
        ({"input_fpr": None}, "input_fpr"),
    ],
)
def test_threshold_gates_fail(metrics: dict[str, float | None], failing: str) -> None:
    out = evaluate(_result(**metrics), parse_suite(SUITE), None, [])
    assert out["status"] == "fail"
    assert [name for name, status in _gates(out).items() if status == "fail"] == [failing]


def test_report_mode_warns_without_failing() -> None:
    out = evaluate(_result(catch_rate=0.8), parse_suite(SUITE), {"metrics": {"catch_rate": 0.95}}, [])
    assert out["status"] == "pass"
    assert _gates(out)["catch_rate"] == "warn"
    no_base = evaluate(_result(catch_rate=0.8), parse_suite(SUITE), None, [])
    assert _gates(no_base)["catch_rate"] == "pass"


def test_report_mode_is_a_gate_when_asked() -> None:
    spec = parse_suite(SUITE.replace("mode: report", "mode: gate"))
    out = evaluate(_result(catch_rate=0.8), spec, {"metrics": {"catch_rate": 0.95}}, [])
    assert out["status"] == "fail"


@pytest.mark.parametrize(
    "text",
    [
        SUITE.replace("{min: 0.85}", "{mode: gate}"),
        SUITE.replace("suite: guardrails", "suite: routing"),
        SUITE.replace("suite: guardrails", "suite: cache"),
        SUITE.replace("{min: 0.85}", "{minimum: 0.85}"),
    ],
)
def test_bad_suite_files_are_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        parse_suite(text)


def test_accepted_entries_need_a_reason() -> None:
    with pytest.raises(ValidationError):
        parse_accepted(ACCEPTED.replace("reason: the new rule is right to block this", "reason: ''"))


def test_markdown_lists_regressions_and_how_to_accept() -> None:
    spec = parse_suite(SUITE)
    failed = evaluate({**_result(), "regressions": [REGRESSION]}, spec, None, [])
    md = render({"guardrails": failed}, base_ref="abc123")
    assert md.startswith("### GG eval gate: FAIL")
    assert "`abc123`" in md
    assert "| guardrails | in-ben-001 | benign | allowed | false_positive |" in md
    assert "accepted_changes.yaml" in md
    assert "input_catch_rate 9/10 = 0.900" in md
    passed = render({"guardrails": evaluate(_result(), spec, None, [])}, base_ref=None)
    assert "Item regressions" not in passed


def test_replay_floor_applies_to_replay_runs_and_min_to_live_ones() -> None:
    spec = parse_suite(SUITE.replace("{min: 0.85}", "{min: 0.85, replay_min: 0.5}"))
    replay = evaluate(_result(input_catch_rate=0.6), spec, None, [])
    live = evaluate({**_result(input_catch_rate=0.6), "mode": "live"}, spec, None, [])
    assert (_gates(replay)["input_catch_rate"], _gates(live)["input_catch_rate"]) == ("pass", "fail")
    assert "Live mode" in render({"guardrails": live}, base_ref=None)


def test_summary_names_guards_that_could_not_decide() -> None:
    out = evaluate({**_result(), "guard_errors": {"jev_injection": 2}}, parse_suite(SUITE), None, [])
    assert "Guards that could not decide: jev_injection on 2 items." in render(
        {"guardrails": out}, base_ref=None
    )
