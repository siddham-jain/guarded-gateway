import pytest

from gg.core.deployment import Deployment, PriceSchedule
from tests.conftest import make_key

PAID = Deployment(id="openai/gpt-6-luna", provider="openai", upstream_model="gpt-6-luna", tier="weak")
FREE = Deployment(
    id="gemini/flash",
    provider="gemini",
    upstream_model="flash",
    pricing=PriceSchedule(periods=(), billed=False),
)
PREMIUM = Deployment(id="anthropic/opus", provider="anthropic", upstream_model="opus", tier="premium")


@pytest.mark.parametrize(
    ("patterns", "model", "allowed"),
    [
        (("gg/*",), "gg/auto", True),
        (("gg/*",), "openai/gpt-6-luna", False),
        (("gg/*", "anthropic/*"), "anthropic/claude-haiku-4-5", True),
        (("openai/gpt-6-*",), "openai/gpt-6-luna", True),
        (("openai/gpt-6-*",), "OPENAI/gpt-6-luna", False),
        ((), "gg/auto", False),
    ],
)
def test_allows_model(patterns: tuple[str, ...], model: str, allowed: bool) -> None:
    assert make_key(allowed_models=patterns).allows_model(model) is allowed


def test_allows_provider() -> None:
    assert make_key().allows_provider("anything")
    key = make_key(allowed_providers=["openai", "anthropic"])
    assert key.allows_provider("openai")
    assert not key.allows_provider("gemini")


def test_allows_deployment_flags() -> None:
    key = make_key()
    assert key.allows_deployment(PAID)
    assert key.allows_deployment(FREE)
    assert not key.allows_deployment(PREMIUM)
    assert make_key(flags={"allow_premium": True}).allows_deployment(PREMIUM)
    assert not make_key(flags={"allow_free_tier_providers": False}).allows_deployment(FREE)
    assert not make_key(allowed_providers=["anthropic"]).allows_deployment(PAID)
