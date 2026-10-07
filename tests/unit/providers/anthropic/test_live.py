"""live s4 check: two-step tool loop on haiku with manual thinking; opt-in with `pytest -m live`"""

from pathlib import Path

import pytest

from gg.config.loader import load_file
from gg.config.settings import Settings
from gg.core.clock import FakeClock
from gg.providers.anthropic.adapter import AnthropicAdapter
from gg.providers.catalog.loader import build_catalog
from gg.providers.catalog.schema import ModelsConfig
from gg.providers.http import HttpClientFactory
from gg.providers.registry import build_adapters
from gg.providers.runtime import AdapterDeps
from gg.providers.state.memory import InMemoryStateStore
from tests.conftest import make_ctx, make_request

pytestmark = pytest.mark.live

MODELS = Path(__file__).resolve().parents[4] / "config" / "models.yaml"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


async def test_haiku_tool_loop_reinjects_thinking() -> None:
    catalog = build_catalog(load_file(MODELS, ModelsConfig), Settings(), profile="prod")
    dep = catalog.get("anthropic/claude-haiku-4-5")
    if not dep.enabled:
        pytest.skip("GG_PROVIDERS__ANTHROPIC__API_KEY is not set")
    http = HttpClientFactory()
    try:
        adapter = build_adapters(catalog, AdapterDeps(http=http, state=InMemoryStateStore()))["anthropic"]
        assert isinstance(adapter, AnthropicAdapter)
        ask = {"role": "user", "content": "What is the weather in Paris? Use the tool."}
        first = make_request(
            model=dep.id, tools=TOOLS, reasoning_effort="low", max_completion_tokens=2048, messages=[ask]
        )
        reply = await adapter.complete(first, dep, make_ctx(FakeClock(), first))
        calls = reply.choices[0].message.tool_calls or ()
        assert reply.choices[0].finish_reason == "tool_calls"
        assert calls
        second = make_request(
            model=dep.id,
            tools=TOOLS,
            reasoning_effort="low",
            max_completion_tokens=2048,
            messages=[
                ask,
                {"role": "assistant", "content": None, "tool_calls": [c.model_dump() for c in calls]},
                *({"role": "tool", "tool_call_id": c.id, "content": "18C and sunny"} for c in calls),
            ],
        )
        final = await adapter.complete(second, dep, make_ctx(FakeClock(), second))
        assert final.choices[0].finish_reason == "stop"
        assert "thinking_reinjected" in final.gg_meta.flags
    finally:
        await http.aclose()
