"""deterministic checkers: a response text -> grade in {0, 1}; judge items return None (graded pairwise)"""

import math
import re
from typing import Any

import orjson

from gg.routing.eval.items import (
    ChoiceGrader,
    ConstraintsGrader,
    ExactGrader,
    Grader,
    JsonGrader,
    NumericGrader,
    RegexGrader,
)

_ANSWER_RE = re.compile(r"(?im)^\W*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$")
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.S)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_SENTENCE_RE = re.compile(r"[.!?]+(?=\s|$)")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def clean(text: str) -> str:
    """drops reasoning blocks that some open models inline in content"""
    return _THINK_RE.sub("", text).strip()


def final_answer(text: str) -> str:
    """the value after the last 'Answer:' line, else the whole text"""
    matches = _ANSWER_RE.findall(clean(text))
    return matches[-1] if matches else clean(text)


def normalize(text: str) -> str:
    text = text.replace("`", "").replace("**", "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".").strip().strip("'\"").strip()


def _unfence(text: str) -> str:
    fenced = _FENCE_RE.findall(text)
    return "\n".join(fenced) if fenced else text


def _number(text: str) -> float | None:
    found = _NUMBER_RE.findall(text.replace("$", "").replace("€", "").replace("£", ""))
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def _numeric(g: NumericGrader, text: str) -> bool:
    value = _number(final_answer(text))
    if value is None:
        return False
    return math.isclose(value, g.answer, abs_tol=g.tol, rel_tol=g.rel_tol)


def _has_word(haystack: str, needle: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(normalize(needle))}(?!\w)", haystack) is not None


def _exact(g: ExactGrader, text: str) -> bool:
    answers = {normalize(a) for a in g.answers}
    if g.match == "word":
        hay = normalize(final_answer(text))
        return any(_has_word(hay, a) for a in g.answers) and not any(_has_word(hay, r) for r in g.reject)
    body = _unfence(clean(text))
    candidates = [final_answer(text), body, *body.splitlines()]
    for candidate in candidates:
        value = normalize(re.sub(r"(?i)^\s*(output|result)\s*:\s*", "", candidate))
        if value in answers:
            return True
    return False


def _choice(g: ChoiceGrader, text: str) -> bool:
    answer = final_answer(text)
    found = re.findall(r"(?<![A-Za-z])\(?([A-H])\)?(?![A-Za-z])", answer)
    return bool(found) and found[0] == g.answer


def _regex(g: RegexGrader, text: str) -> bool:
    flags = 0 if g.case_sensitive else re.I
    body = clean(text)
    return all(re.search(p, body, flags) for p in g.all) and not any(
        re.search(p, body, flags) for p in g.none
    )


def parse_json(text: str) -> Any:
    """first json value in the text, fences allowed"""
    body = _unfence(clean(text)).strip()
    try:
        return orjson.loads(body)
    except orjson.JSONDecodeError:
        pass
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(body.rfind("}"), body.rfind("]"))
    try:
        return orjson.loads(body[start : end + 1])
    except orjson.JSONDecodeError:
        return None


def json_matches(expect: Any, actual: Any, tol: float) -> bool:
    if isinstance(expect, dict) and "$any" in expect:
        return any(json_matches(e, actual, tol) for e in expect["$any"])  # pyright: ignore[reportUnknownVariableType]
    if isinstance(expect, dict) and "$contains" in expect:
        return isinstance(actual, str) and normalize(str(expect["$contains"])) in normalize(actual)
    if isinstance(expect, dict):
        return isinstance(actual, dict) and all(
            k in actual and json_matches(v, actual[k], tol)  # pyright: ignore[reportUnknownArgumentType]
            for k, v in expect.items()  # pyright: ignore[reportUnknownVariableType]
        )
    if isinstance(expect, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expect)  # pyright: ignore[reportUnknownArgumentType]
            and all(json_matches(e, a, tol) for e, a in zip(expect, actual, strict=True))  # pyright: ignore[reportUnknownArgumentType]
        )
    if isinstance(expect, bool) or expect is None:
        return actual is expect
    if isinstance(expect, (int, float)):
        if isinstance(actual, str):
            actual = _number(actual)
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(actual, expect, abs_tol=tol)
        )
    return isinstance(actual, str) and normalize(actual) == normalize(str(expect))


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'\u2019-]+", text)


def _constraints(g: ConstraintsGrader, text: str) -> bool:
    body = clean(text)
    lower = body.lower()
    words = _words(body)
    lines = [line for line in body.splitlines() if line.strip()]
    checks = [
        g.max_words is None or len(words) <= g.max_words,
        g.min_words is None or len(words) >= g.min_words,
        g.exact_words is None or len(words) == g.exact_words,
        g.max_sentences is None or len(_SENTENCE_RE.findall(body)) <= g.max_sentences,
        g.lines is None or len(lines) == g.lines,
        g.max_lines is None or len(lines) <= g.max_lines,
        g.list_items is None or sum(1 for line in lines if _LIST_ITEM_RE.match(line)) == g.list_items,
        g.word_initial is None or all(w.lower().startswith(g.word_initial.lower()) for w in words),
        all(s.lower() in lower for s in g.contains_all),
        not g.contains_any or any(s.lower() in lower for s in g.contains_any),
        not any(s.lower() in lower for s in g.contains_none),
        all(re.search(p, body, re.I) for p in g.regex),
    ]
    return all(checks)


def grade(grader: Grader, text: str) -> float | None:
    """1.0 pass, 0.0 fail; None means the item is graded by the pairwise judge"""
    match grader:
        case NumericGrader():
            ok = _numeric(grader, text)
        case ExactGrader():
            ok = _exact(grader, text)
        case ChoiceGrader():
            ok = _choice(grader, text)
        case RegexGrader():
            ok = _regex(grader, text)
        case JsonGrader():
            parsed = parse_json(text)
            ok = parsed is not None and json_matches(grader.expect, parsed, grader.tol)
        case ConstraintsGrader():
            ok = _constraints(grader, text)
        case _:
            return None
    return 1.0 if ok else 0.0
