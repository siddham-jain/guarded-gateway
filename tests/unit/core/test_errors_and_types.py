from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from gg.core.deployment import PriceSchedule, Pricing
from gg.core.errors import (
    GGError,
    GuardrailBlockedError,
    GuardrailUnavailableError,
    QuotaExceededError,
    RateLimitedError,
    UpstreamError,
)
from gg.core.jsonutil import canonical_json, dumps
from gg.core.log import mask_secrets
from gg.core.vault import PlaceholderVault


@pytest.mark.parametrize(
    ("error", "status", "code", "retry"),
    [
        (GuardrailBlockedError("blocked"), 400, "guardrail_blocked", "false"),
        (GuardrailUnavailableError("down"), 503, "guardrail_unavailable", "false"),
        (QuotaExceededError("over"), 402, "budget_exceeded", "false"),
        (RateLimitedError("slow", retry_after_s=2.4), 429, "rate_limit_exceeded", "true"),
        (UpstreamError("bad"), 502, "upstream_error", "false"),
    ],
)
def test_error_shape(error: GGError, status: int, code: str, retry: str) -> None:
    body = error.to_body()
    assert error.status == status
    assert body["error"]["code"] == code
    assert set(body["error"]) >= {"message", "type", "param", "code"}
    assert error.response_headers()["x-should-retry"] == retry


def test_retry_after_header_is_whole_seconds() -> None:
    assert RateLimitedError("x", retry_after_s=2.4).response_headers()["retry-after"] == "2"
    assert RateLimitedError("x", retry_after_s=0.1).response_headers()["retry-after"] == "1"


def test_vault_reuses_placeholders_and_hides_values() -> None:
    vault = PlaceholderVault()
    a = vault.add("EMAIL", "a@x.com")
    b = vault.add("EMAIL", "b@x.com")
    assert (a, b) == ("<EMAIL_1>", "<EMAIL_2>")
    assert vault.add("EMAIL", "a@x.com") == a
    assert vault.resolve(a) == "a@x.com"
    assert "a@x.com" not in repr(vault)


def test_price_schedule_picks_latest_effective_period() -> None:
    schedule = PriceSchedule(
        periods=(
            Pricing(effective_from=date(2026, 1, 1), input=Decimal(1), output=Decimal(2)),
            Pricing(effective_from=date(2027, 1, 1), input=Decimal(2), output=Decimal(4)),
        )
    )
    current = schedule.at(datetime(2026, 10, 5, tzinfo=UTC))
    assert current is not None
    assert current.input == Decimal(1)
    assert schedule.at(datetime(2025, 1, 1, tzinfo=UTC)) is None


def test_json_handles_decimal_and_sorts_canonically() -> None:
    assert dumps({"x": Decimal("1.50")}) == b'{"x":"1.50"}'
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


@pytest.mark.parametrize(
    "raw",
    [
        "key gg-live-abcdefghijk",
        "sk-proj-abcdefghijklmnop",
        "Authorization: Bearer abc.def",
        "hf_abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_secrets_are_masked_in_logs(raw: str) -> None:
    masked = mask_secrets(raw)
    assert "****" in masked
    assert raw != masked
