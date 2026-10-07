from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from gg.routing.eval.graders import final_answer, grade, json_matches, parse_json
from gg.routing.eval.items import EvalItem, limit_items, load_items, load_suite
from gg.routing.eval.judge import JudgePrompt, combine, grades_for, parse_verdict, winner_of

ROOT = Path(__file__).resolve().parents[4]
SUITE = ROOT / "evals/routing/suite.yaml"
ITEMS = load_items(load_suite(SUITE).items)


def make_item(grader: dict[str, Any], **extra: Any) -> EvalItem:
    return EvalItem.model_validate(
        {
            "id": "rt999",
            "category": "test",
            "split": "tune",
            "expected_tier": "weak",
            "messages": [{"role": "user", "content": "q"}],
            "grader": grader,
            **extra,
        }
    )


def test_dataset_shape() -> None:
    assert 120 <= len(ITEMS) <= 200
    assert len({i.id for i in ITEMS}) == len(ITEMS)
    tiers = Counter(i.expected_tier for i in ITEMS)
    assert 0.35 <= tiers["strong"] / len(ITEMS) <= 0.65
    heldout = sum(i.split == "heldout" for i in ITEMS) / len(ITEMS)
    assert 0.3 <= heldout <= 0.5
    # most items grade without a judge
    assert sum(i.needs_judge for i in ITEMS) / len(ITEMS) < 0.2


def test_every_reference_passes_its_own_checker() -> None:
    failing = [
        i.id
        for i in ITEMS
        if not i.needs_judge and (i.reference is None or grade(i.grader, i.reference) != 1)
    ]
    assert failing == []


def test_suite_pairs_resolve_paths() -> None:
    suite = load_suite(SUITE)
    assert suite.judge_prompt.is_file()
    assert {"ci", "dev-free", "openrouter"} <= set(suite.pairs)
    assert suite.pairs["ci"].profile == "ci"


def test_limit_is_round_robin_over_categories() -> None:
    picked = limit_items(ITEMS, 13)
    assert len(picked) == 13
    assert len({i.category for i in picked}) == 13
    assert limit_items(ITEMS, None) == ITEMS


def test_final_answer_takes_the_last_answer_line() -> None:
    assert final_answer("work...\nAnswer: 3\nmore\nFinal answer: 42") == "42"
    assert final_answer("just text") == "just text"


@pytest.mark.parametrize(
    ("grader", "text", "expected"),
    [
        ({"type": "numeric", "answer": 2835.08, "tol": 0.005}, "so...\nAnswer: $2,835.08", 1.0),
        ({"type": "numeric", "answer": 12}, "Answer: 13", 0.0),
        (
            {"type": "numeric", "answer": 7.5, "tol": 0.01},
            "<think>maybe 8</think>The angle is 7.5 degrees",
            1.0,
        ),
        ({"type": "exact", "answers": ["[1, 2, 3]"]}, "```\n[1, 2, 3]\n```", 1.0),
        ({"type": "exact", "answers": ["edc"]}, "Output: edc", 1.0),
        ({"type": "exact", "answers": ["2 4"]}, "24", 0.0),
        (
            {"type": "exact", "answers": ["Canberra"], "match": "word", "reject": ["Sydney"]},
            "It's Canberra.",
            1.0,
        ),
        (
            {"type": "exact", "answers": ["Canberra"], "match": "word", "reject": ["Sydney"]},
            "Sydney or Canberra",
            0.0,
        ),
        ({"type": "choice", "answer": "C"}, "Reasoning...\nAnswer: C) Serializable", 1.0),
        ({"type": "choice", "answer": "C"}, "Answer: B", 0.0),
        ({"type": "regex", "all": [r"git (switch -c|checkout -b) feature"]}, "`git switch -c feature`", 1.0),
        ({"type": "regex", "all": ["Lazy Dog"], "case_sensitive": True}, "lazy dog", 0.0),
        (
            {"type": "json", "expect": {"a": 1.5, "b": {"$any": ["x", "y"]}}},
            '```json\n{"a": "1.5", "b": "Y"}\n```',
            1.0,
        ),
        ({"type": "json", "expect": [1, 2]}, "[2, 1]", 0.0),
        ({"type": "constraints", "list_items": 3}, "1. a\n2. b\n3. c", 1.0),
        ({"type": "constraints", "max_words": 3}, "one two three four", 0.0),
        ({"type": "constraints", "exact_words": 4, "word_initial": "s"}, "Silver seas sway softly.", 1.0),
        ({"type": "constraints", "contains_any": ["please"], "contains_none": ["now!"]}, "Send it now!", 0.0),
        ({"type": "judge"}, "anything", None),
    ],
)
def test_graders(grader: dict[str, Any], text: str, expected: float | None) -> None:
    assert grade(make_item(grader).grader, text) == expected


def test_parse_json_and_contains() -> None:
    assert parse_json('Sure! {"x": [1, 2]} hope that helps') == {"x": [1, 2]}
    assert parse_json("no json here") is None
    assert json_matches({"name": {"$contains": "aurora x2"}}, {"name": "The Aurora X2 headphones"}, 0.01)
    assert not json_matches({"flag": False}, {"flag": 0}, 0.01)


def test_parse_verdict() -> None:
    assert parse_verdict('{"verdict": "A", "reason": "x"}') == "A"
    assert parse_verdict('```json\n{"verdict": "tie"}\n```') == "tie"
    assert parse_verdict("after thought, Verdict: B") == "B"
    assert parse_verdict("no idea") == "invalid"


def test_orders_map_to_models_and_disagreement_is_a_tie() -> None:
    # ws: answer A is weak's; sw: answer A is strong's
    assert winner_of("ws", "A") == "weak"
    assert winner_of("sw", "A") == "strong"
    assert combine("B", "A") == ("strong", True)
    assert combine("A", "A") == ("tie", False)
    assert combine("invalid", "A") == ("tie", False)
    assert grades_for("strong") == (0.0, 1.0)
    assert grades_for("tie") == (0.5, 0.5)


def test_judge_prompt_render_does_not_expand_placeholders_inside_answers() -> None:
    prompt = JudgePrompt.load(load_suite(SUITE).judge_prompt)
    item = make_item({"type": "judge", "rubric": "be right"}, reference="ref")
    text = prompt.render(item, "A says {{answer_b}}", "B")
    assert "A says {{answer_b}}" in text
    assert "be right" in text
    assert "ref" in text
    assert prompt.version.startswith("pairwise_v1:")
