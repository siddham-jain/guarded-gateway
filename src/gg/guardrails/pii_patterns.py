"""regex pii recognizers with checksums; the baseline the presidio guard augments"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from gg.guardrails.secrets_rules import trailing_run
from gg.guardrails.vault import LABELS, canonical


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    compact = value.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _card_ok(value: str) -> bool:
    digits = _digits(value)
    return 13 <= len(digits) <= 19 and luhn_ok(digits)


def _phone_ok(value: str) -> bool:
    return 8 <= len(_digits(value)) <= 15


_DIGIT_TAIL_RE = re.compile(r"\d[\d ().+-]*$")


@dataclass(frozen=True, slots=True)
class Recognizer:
    entity: str
    regex: re.Pattern[str]
    score: float
    validate: Callable[[str], bool] | None = None


RECOGNIZERS: tuple[Recognizer, ...] = (
    Recognizer(
        "EMAIL_ADDRESS",
        re.compile(
            r"(?<![\w.%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}(?![\w-])"
        ),
        1.0,
    ),
    Recognizer("CREDIT_CARD", re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"), 1.0, _card_ok),
    Recognizer(
        "IBAN_CODE",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}(?![A-Za-z0-9])"),
        1.0,
        iban_ok,
    ),
    Recognizer(
        "US_SSN", re.compile(r"(?<![\d-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\d-])"), 0.85
    ),
    Recognizer(
        "PHONE_NUMBER",
        re.compile(
            r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?!\w)"
            r"|(?<!\w)\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)"
        ),
        0.7,
        _phone_ok,
    ),
    Recognizer(
        "IP_ADDRESS",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\.?\d)"
        ),
        0.6,
    ),
)

ENTITIES: frozenset[str] = frozenset(r.entity for r in RECOGNIZERS)


@dataclass(frozen=True, slots=True)
class PiiMatch:
    start: int
    end: int
    entity: str
    label: str
    score: float


class PiiDetector:
    def __init__(
        self, entities: Iterable[str], *, allow_values: Iterable[str] = (), min_score: float = 0.0
    ) -> None:
        wanted = set(entities)
        unknown = wanted - ENTITIES
        if unknown:
            raise ValueError(f"unsupported pii entities: {', '.join(sorted(unknown))}")
        self._recognizers = tuple(r for r in RECOGNIZERS if r.entity in wanted and r.score >= min_score)
        self._allow = {
            (LABELS[r.entity], canonical(LABELS[r.entity], v)) for v in allow_values for r in RECOGNIZERS
        }

    def find(self, text: str) -> list[PiiMatch]:
        found: list[PiiMatch] = []
        for rec in self._recognizers:
            label = LABELS[rec.entity]
            for m in rec.regex.finditer(text):
                value = m.group(0)
                if rec.validate is not None and not rec.validate(value):
                    continue
                if (label, canonical(label, value)) in self._allow:
                    continue
                found.append(PiiMatch(m.start(), m.end(), rec.entity, label, rec.score))
        return sorted(found, key=lambda p: p.start)

    @staticmethod
    def holdback(text: str) -> int:
        # an unfinished email or a digit group that may still grow into a phone or card number
        tail = text[-40:]
        m = _DIGIT_TAIL_RE.search(tail)
        digits = len(tail) - m.start() if m is not None and len(_digits(m.group(0))) >= 3 else 0
        return max(digits, trailing_run(text))
