"""shared secret detector: the `secrets` input guard and the `secrets_out` output guard"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from gg.core.schema import StrictModel
from gg.guardrails.rules import RUN_CONTINUATION, SecretsPack

_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{20,}")
# shapes that look random but are not secrets: uuids, git shas, sha256 digests
_EXCLUDED_SHAPES = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$"
)
_PEM_PARTIAL_RE = re.compile(r"-{1,5}(?:B(?:E(?:G(?:I(?:N[^\n]{0,40})?)?)?)?)?$")


class EntropyCfg(StrictModel):
    enabled: bool = True
    threshold: Annotated[float, Field(ge=0, le=8)] = 4.5
    min_len: Annotated[int, Field(ge=8)] = 20


def shannon_entropy(value: str) -> float:
    counts = Counter(value)
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def trailing_run(text: str, limit: int = 128) -> int:
    """length of the trailing non-whitespace run, scanning at most `limit` chars back"""
    n = 0
    for ch in reversed(text[-limit:]):
        if ch.isspace():
            break
        n += 1
    return n


@dataclass(frozen=True, slots=True)
class SecretMatch:
    start: int
    end: int
    rule: str
    continuation: str


class SecretDetector:
    def __init__(self, pack: SecretsPack, entropy: EntropyCfg) -> None:
        self._pack = pack
        self._entropy = entropy
        self._window = pack.doc.keyword_window
        self._allow = pack.doc.allow_substrings

    def _near(self, keywords: re.Pattern[str], text: str, start: int) -> bool:
        return keywords.search(text, max(0, start - self._window), start) is not None

    def _allowed(self, value: str) -> bool:
        return any(a in value for a in self._allow)

    def find(self, text: str) -> list[SecretMatch]:
        found: list[SecretMatch] = []
        for rule in self._pack.rules:
            for m in rule.regex.finditer(text):
                start, end = m.span(rule.group)
                if start == end or self._allowed(m.group(rule.group)):
                    continue
                if rule.keywords is not None and not self._near(rule.keywords, text, m.start()):
                    continue
                found.append(SecretMatch(start, end, rule.id, rule.continuation))
        keywords = self._pack.entropy_keywords
        if self._entropy.enabled and keywords is not None:
            for m in _TOKEN_RE.finditer(text):
                value = m.group(0)
                if (
                    len(value) < self._entropy.min_len
                    or _EXCLUDED_SHAPES.match(value)
                    or self._allowed(value)
                    or any(s.start < m.end() and m.start() < s.end for s in found)
                ):
                    continue
                if shannon_entropy(value) >= self._entropy.threshold and self._near(
                    keywords, text, m.start()
                ):
                    found.append(SecretMatch(m.start(), m.end(), "entropy", RUN_CONTINUATION))
        return sorted(found, key=lambda s: s.start)

    @staticmethod
    def holdback(text: str) -> int:
        # an unfinished token at the end could be the start of a key; so could an unclosed pem header
        hold = trailing_run(text)
        pem = _PEM_PARTIAL_RE.search(text[-64:])
        if pem is not None:
            hold = max(hold, len(pem.group(0)))
        return hold
