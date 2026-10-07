"""pairwise judge (C5 §10.6): both orders, an agreed verdict wins, disagreement is a tie"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gg.core.jsonutil import sha256_hex
from gg.routing.eval.graders import clean, parse_json
from gg.routing.eval.items import EvalItem

type Verdict = Literal["A", "B", "tie", "invalid"]
type Winner = Literal["weak", "strong", "tie"]
type Order = Literal["ws", "sw"]

# ws: answer A is the weak model's; sw: answer A is the strong model's
ORDERS: tuple[Order, ...] = ("ws", "sw")
_VERDICT_RE = re.compile(r"verdict\W{0,5}(A|B|tie)\b", re.I)


@dataclass(frozen=True, slots=True)
class JudgePrompt:
    template: str
    version: str

    @classmethod
    def load(cls, path: Path) -> "JudgePrompt":
        template = path.read_text()
        for slot in ("{{conversation}}", "{{answer_a}}", "{{answer_b}}"):
            if slot not in template:
                raise ValueError(f"{path}: judge prompt is missing {slot}")
        return cls(template, f"{path.stem}:{sha256_hex(template)[:8]}")

    def render(self, item: EvalItem, answer_a: str, answer_b: str) -> str:
        conversation = "\n\n".join(f"[{m.role}]\n{m.content}" for m in item.messages)
        reference = item.reference or "(none)"
        rubric = (item.grader.rubric if item.grader.type == "judge" else None) or "(none)"
        values = {
            "{{conversation}}": conversation,
            "{{reference}}": reference,
            "{{rubric}}": rubric,
            "{{answer_a}}": clean(answer_a) or "(empty answer)",
            "{{answer_b}}": clean(answer_b) or "(empty answer)",
        }
        # one pass so a placeholder inside an answer is never expanded
        return re.sub(r"\{\{\w+\}\}", lambda m: values.get(m.group(0), m.group(0)), self.template)


def parse_verdict(text: str) -> Verdict:
    parsed = parse_json(text)
    if isinstance(parsed, dict):
        value = str(parsed.get("verdict", "")).strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if value.lower() == "tie":
            return "tie"
        if value.upper() in ("A", "B"):
            return "A" if value.upper() == "A" else "B"
    found = _VERDICT_RE.findall(clean(text))
    if found:
        last = found[-1]
        return "tie" if last.lower() == "tie" else ("A" if last.upper() == "A" else "B")
    return "invalid"


def winner_of(order: Order, verdict: Verdict) -> Winner | None:
    if verdict == "invalid":
        return None
    if verdict == "tie":
        return "tie"
    a_is_weak = order == "ws"
    return "weak" if (verdict == "A") == a_is_weak else "strong"


def combine(ws: Verdict, sw: Verdict) -> tuple[Winner, bool]:
    """final winner and whether both orders agreed; an invalid or disagreeing pair is a tie"""
    first, second = winner_of("ws", ws), winner_of("sw", sw)
    if first is None or second is None or first != second:
        return "tie", False
    return first, True


def grades_for(winner: Winner) -> tuple[float, float]:
    """(weak grade, strong grade)"""
    if winner == "strong":
        return 0.0, 1.0
    if winner == "weak":
        return 1.0, 0.0
    return 0.5, 0.5
