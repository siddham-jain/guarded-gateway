from collections.abc import Callable, Iterable, Mapping

OTHER = "other"
NONE = "none"

# labels whose values come from config or callers; everything else is a closed enum built in code
DEFAULT_ALLOWED: Mapping[str, frozenset[str]] = {
    "provider": frozenset({"openai", "anthropic", "gemini", "ollama", "mock", "cache", NONE}),
    "tier": frozenset({"weak", "strong", "premium", NONE}),
    "stage": frozenset(
        {
            "observability",
            "limits",
            "guard_in_pre",
            "guard_out",
            "cache_exact",
            "probes",
            "routing",
            "terminal",
        }
    ),
    "alias": frozenset(),
    "deployment": frozenset({NONE}),
    "error_kind": frozenset(
        {"retryable", "fallback", "client", "auth", "quota_minute", "quota_day", "content_filter", NONE}
    ),
    "error_type": frozenset(
        {
            "invalid_request",
            "auth",
            "not_found",
            "rate_limited",
            "budget_exceeded",
            "guardrail_blocked",
            "upstream_unavailable",
            "upstream_midstream",
            "timeout",
            "internal",
        }
    ),
    "action": frozenset({"allow", "flag", "redact", "block", "skip"}),
    "mode": frozenset({"enforce", "shadow", "off"}),
    "direction": frozenset({"redacted", "restored", "unmatched", "restore_failed"}),
    "strategy": frozenset({"none", "exact", "case_insensitive"}),
    "token_type": frozenset({"input", "cached_input", "cache_write", "output", "reasoning"}),
    "breaker_reason": frozenset(
        {
            "trip",
            "cooldown_elapsed",
            "probe_ok",
            "probe_fail",
            "auth",
            "per_day",
            "rate_limited",
            "model_not_found",
        }
    ),
    "guardrail": frozenset(),
    "cache_reason": frozenset(),
    "error_code": frozenset(),
}

# labels fed by code-defined strings that are not typed enums: the first values seen are kept up to the cap
DEFAULT_CAPS: Mapping[str, int] = {
    "strategy": 8,
    "breaker_reason": 24,
    "guardrail": 32,
    "cache_reason": 48,
    "error_code": 32,
}

_MAX_REPORTED = 1000
_MAX_VALUE_LEN = 64


class LabelGuard:
    """maps label values outside the allowlist to 'other' so series stay bounded"""

    def __init__(self, on_unknown: Callable[[str, str], None]) -> None:
        self._allowed: dict[str, set[str]] = {name: set(values) for name, values in DEFAULT_ALLOWED.items()}
        self._caps: dict[str, int] = dict(DEFAULT_CAPS)
        self._on_unknown = on_unknown
        self._reported: set[tuple[str, str]] = set()

    def allow(self, label: str, values: Iterable[str]) -> None:
        self._allowed.setdefault(label, set()).update(values)

    def __call__(self, label: str, value: str | None) -> str:
        if value is None:
            return NONE
        allowed = self._allowed.get(label)
        if allowed is None:
            raise KeyError(f"label {label!r} has no allowlist")
        if value in allowed:
            return value
        cap = self._caps.get(label)
        if cap is not None and len(allowed) < cap and len(value) <= _MAX_VALUE_LEN:
            allowed.add(value)
            return value
        key = (label, value)
        if key not in self._reported and len(self._reported) < _MAX_REPORTED:
            self._reported.add(key)
            self._on_unknown(label, value)
        return OTHER
