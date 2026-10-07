import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NoReturn

from gg.core.vault import PlaceholderVault

PLACEHOLDER_RE = re.compile(r"\[([A-Z]{2,12})_(\d{1,4})\]")
_LOOSE_PLACEHOLDER_RE = re.compile(r"\[([A-Za-z]{2,12})_(\d{1,4})\]")
_PARTIAL_RE = re.compile(r"\[(?:[A-Za-z]{1,12}(?:_\d{0,4})?)?$")

# presidio entity names -> placeholder labels; also the bounded label set for metrics
LABELS: dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "IBAN_CODE": "IBAN",
    "US_SSN": "SSN",
    "IP_ADDRESS": "IP",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "SECRET": "SECRET",
}


def canonical(label: str, value: str) -> str:
    """trivial variants of one value share a placeholder"""
    if label == "EMAIL":
        return value.strip().lower()
    if label in ("PHONE", "CARD", "SSN"):
        return "".join(ch for ch in value if ch.isdigit())
    if label == "IBAN":
        return value.replace(" ", "").upper()
    return value


@dataclass(slots=True)
class RestoreStats:
    exact: int = 0
    case_insensitive: int = 0
    unmatched: int = 0


class GuardVault(PlaceholderVault):
    """placeholders look like [EMAIL_1]; numbering skips placeholder-shaped literals the client typed"""

    def __init__(self) -> None:
        super().__init__()
        self._reserved: set[str] = set()
        self._canonical: dict[tuple[str, str], str] = {}
        self.stats = RestoreStats()

    @classmethod
    def adopt(cls, vault: PlaceholderVault, texts: Iterable[str] = ()) -> "GuardVault":
        out = vault if isinstance(vault, GuardVault) else cls()
        out.reserve(texts)
        if out is not vault:
            for placeholder, value in vault.placeholders().items():
                out._by_placeholder[placeholder] = value
        return out

    def reserve(self, texts: Iterable[str]) -> None:
        for text in texts:
            for m in _LOOSE_PLACEHOLDER_RE.finditer(text):
                self._reserved.add(m.group(0).upper())

    def add(self, label: str, value: str) -> str:
        key = (label, canonical(label, value))
        existing = self._canonical.get(key)
        if existing is not None:
            return existing
        n = self._counts.get(label, 0)
        while True:
            n += 1
            placeholder = f"[{label}_{n}]"
            if placeholder not in self._reserved:
                break
        self._counts[label] = n
        self._canonical[key] = placeholder
        self._by_value[(label, value)] = placeholder
        self._by_placeholder[placeholder] = value
        return placeholder

    def is_vault_value(self, label: str, value: str) -> bool:
        return (label, canonical(label, value)) in self._canonical

    def restore(self, text: str, *, json_escape: bool = False) -> str:
        if not self._by_placeholder or "[" not in text:
            return text

        def sub(m: re.Match[str]) -> str:
            token = m.group(0)
            value = self._by_placeholder.get(token)
            if value is not None:
                self.stats.exact += 1
            else:
                upper = token.upper()
                value = None if upper in self._reserved else self._by_placeholder.get(upper)
                if value is None:
                    if PLACEHOLDER_RE.fullmatch(token):
                        self.stats.unmatched += 1
                    return token
                self.stats.case_insensitive += 1
            return json.dumps(value)[1:-1] if json_escape else value

        return _LOOSE_PLACEHOLDER_RE.sub(sub, text)

    @staticmethod
    def partial_suffix_len(text: str) -> int:
        m = _PARTIAL_RE.search(text[-20:])
        return len(m.group(0)) if m else 0

    def summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for placeholder in self._by_placeholder:
            m = PLACEHOLDER_RE.fullmatch(placeholder)
            label = m.group(1) if m else "OTHER"
            out[label] = out.get(label, 0) + 1
        return out

    def __reduce__(self) -> NoReturn:
        raise TypeError("PlaceholderVault must never be serialised")

    def __repr__(self) -> str:
        return f"GuardVault({self.summary()})"
