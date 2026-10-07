from datetime import date

from pydantic import SecretStr

from gg.config.settings import ProviderCredentials, Settings
from gg.providers.catalog.loader import build_catalog
from tests.conftest import make_request
from tests.unit.providers.support import models_config

TODAY = date(2026, 10, 5)


def _chain(**keys: str) -> list[str]:
    providers = {name: ProviderCredentials(api_key=SecretStr(key)) for name, key in keys.items()}
    settings = Settings(_env_file=None, providers=providers)  # pyright: ignore[reportCallIssue]
    catalog = build_catalog(models_config(), settings, profile="demo", today=TODAY)
    return [d.id for d in catalog.chain("demo-chaos", make_request())]


def test_demo_chaos_falls_over_from_the_failing_mock_to_gemini() -> None:
    assert _chain(gemini="test-key") == ["mock/down", "gemini/gemini-3.1-flash-lite", "mock/echo"]


def test_demo_chaos_still_serves_without_provider_keys() -> None:
    assert _chain() == ["mock/down", "mock/echo"]
