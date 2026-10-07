from collections.abc import Mapping
from types import MappingProxyType


class PlaceholderVault:
    """per-request reversible placeholder map; never logged, never cached"""

    def __init__(self) -> None:
        self._by_placeholder: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def add(self, label: str, value: str) -> str:
        existing = self._by_value.get((label, value))
        if existing is not None:
            return existing
        n = self._counts.get(label, 0) + 1
        self._counts[label] = n
        placeholder = f"<{label}_{n}>"
        self._by_placeholder[placeholder] = value
        self._by_value[(label, value)] = placeholder
        return placeholder

    def resolve(self, placeholder: str) -> str | None:
        return self._by_placeholder.get(placeholder)

    def placeholders(self) -> Mapping[str, str]:
        return MappingProxyType(self._by_placeholder)

    def __len__(self) -> int:
        return len(self._by_placeholder)

    def __repr__(self) -> str:
        return f"PlaceholderVault(n={len(self._by_placeholder)})"
