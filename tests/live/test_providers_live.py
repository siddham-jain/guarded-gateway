"""live smoke per provider (c3 §8.5); opt-in with `pytest -m live`, skipped without keys"""

from pathlib import Path

import pytest

from gg.config.loader import load_file
from gg.config.settings import Settings
from gg.core.clock import FakeClock
from gg.providers.catalog.loader import build_catalog
from gg.providers.catalog.schema import ModelsConfig
from gg.providers.http import HttpClientFactory
from gg.providers.registry import build_adapters
from gg.providers.runtime import AdapterDeps
from tests.conftest import make_ctx, make_request

pytestmark = pytest.mark.live

MODELS = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
CASES = [
    "openai/gpt-6-luna",
    "groq/openai/gpt-oss-120b",
    "deepinfra/openai/gpt-oss-120b",
    "cerebras/openai/gpt-oss-120b",
    "fireworks/openai/gpt-oss-120b",
    "deepseek/deepseek-flash",
    "zai/glm-4.7-flash",
    "qwen/qwen3.8-flash",
    "mistral/mistral-small-latest",
    "ollama/llama3.2:1b",
]


@pytest.mark.parametrize("deployment_id", CASES)
async def test_reply_ok(deployment_id: str) -> None:
    catalog = build_catalog(load_file(MODELS, ModelsConfig), Settings(), profile="prod")
    dep = catalog.get(deployment_id)
    if not dep.enabled:
        pytest.skip(f"no credentials for {dep.provider}")
    http = HttpClientFactory()
    try:
        adapter = build_adapters(catalog, AdapterDeps(http=http))[dep.provider]
        request = make_request(
            model=deployment_id,
            max_completion_tokens=16,
            temperature=0,
            messages=[{"role": "user", "content": "Reply with OK"}],
        )
        response = await adapter.complete(request, dep, make_ctx(FakeClock(), request))  # pyright: ignore[reportAttributeAccessIssue]
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
    finally:
        await http.aclose()
