"""live gemini checks for the unverified wire details; opt-in with `pytest -m live`, skipped without a key"""

import os

import pytest

from gg.config.settings import Settings
from gg.core.clock import FakeClock
from gg.providers.catalog.loader import build_catalog
from gg.providers.gemini.adapter import GeminiAdapter
from gg.providers.http import HttpClientFactory
from gg.providers.runtime import AdapterDeps
from gg.providers.state.memory import InMemoryStateStore
from tests.conftest import make_ctx, make_request
from tests.unit.providers.support import models_config

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GG_PROVIDERS__GEMINI__API_KEY"), reason="no gemini key"),
]
DEPLOYMENT = "gemini/gemini-3.1-flash-lite"
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


async def test_text_and_two_step_tool_loop() -> None:
    catalog = build_catalog(models_config(), Settings(), profile="dev-free")
    dep = catalog.get(DEPLOYMENT)
    http = HttpClientFactory()
    adapter = GeminiAdapter(catalog.providers["gemini"], AdapterDeps(http=http, state=InMemoryStateStore()))
    try:
        hello = make_request(
            max_completion_tokens=16, messages=[{"role": "user", "content": "Reply with OK"}]
        )
        reply = await adapter.complete(hello, dep, make_ctx(FakeClock(), hello))
        assert reply.choices[0].finish_reason in ("stop", "length")
        assert reply.usage is not None
        assert reply.usage.prompt_tokens > 0

        ask = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
        first = make_request(tools=TOOLS, tool_choice="required", messages=ask)
        step = await adapter.complete(first, dep, make_ctx(FakeClock(), first))
        calls = step.choices[0].message.tool_calls or ()
        assert calls
        history = [
            *ask,
            {"role": "assistant", "content": None, "tool_calls": [c.model_dump() for c in calls]},
            *({"role": "tool", "tool_call_id": c.id, "content": '{"temp_c": 18}'} for c in calls),
        ]
        second = make_request(tools=TOOLS, messages=history)
        final = await adapter.complete(second, dep, make_ctx(FakeClock(), second))
        assert final.choices[0].message.content

        # cross-provider history: ids gemini never issued, so the dummy signature must be accepted
        foreign = [
            *ask,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_01abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_01abc", "content": '{"temp_c": 18}'},
        ]
        third = make_request(tools=TOOLS, messages=foreign)
        crossed = await adapter.complete(third, dep, make_ctx(FakeClock(), third))
        assert crossed.choices[0].message.content
        assert "dummy_signature" in crossed.gg_meta.flags
    finally:
        await http.aclose()
