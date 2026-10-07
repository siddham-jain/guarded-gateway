import hashlib
from typing import Any

import pytest

from gg.core.errors import ProviderError
from gg.core.schema import ChatRequest
from gg.providers.openai_compat.quirks import QuirkProfile
from gg.providers.openai_compat.request import RequestInfo, Translated, build_body, history_key, history_keys
from tests.conftest import make_request
from tests.unit.providers.support import make_dep, profile_quirks

INFO = RequestInfo(request_id="req_1", key_id="key-a", received_unix=1)
TOOLS = [{"type": "function", "function": {"name": "f", "strict": True, "parameters": {"type": "object"}}}]
TOOL_HISTORY = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call_abc", "content": "sunny"},
]


def build(
    provider: str | dict[str, Any],
    request: ChatRequest | None = None,
    *,
    info: RequestInfo = INFO,
    history: dict[str, str] | None = None,
    **dep: Any,
) -> Translated:
    quirks = QuirkProfile.model_validate(profile_quirks(provider) if isinstance(provider, str) else provider)
    return build_body(request or make_request(), make_dep(**dep), quirks, info, history=history)


def test_openai_is_near_passthrough() -> None:
    req = make_request(
        model="openai/gpt-6-luna",
        max_completion_tokens=50,
        user="alice",
        verbosity="low",
        service_tier="priority",
        gg={"cache": "off"},
        temperature=0.5,
    )
    out = build("openai", req, model="gpt-6-luna")
    body = out.body
    assert body["model"] == "gpt-6-luna"
    assert body["max_completion_tokens"] == 50
    assert "max_tokens" not in body
    assert body["verbosity"] == "low"
    assert "service_tier" not in body
    assert "service_tier" in out.ignored
    assert "gg" not in body
    assert "user" not in body
    assert body["safety_identifier"] == hashlib.sha256(b"key-a:alice").hexdigest()
    assert body["store"] is False
    assert body["prompt_cache_key"] == hashlib.sha256(b"key-a").hexdigest()[:32]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True, "include_obfuscation": False}


def test_openai_store_and_service_tier_allowed_by_key_flags() -> None:
    info = RequestInfo(request_id="r", key_id="k", received_unix=1, allow_store=True, allow_service_tier=True)
    body = build("openai", make_request(store=True, service_tier="flex"), info=info).body
    assert body["store"] is True
    assert body["service_tier"] == "flex"


def test_other_hosts_drop_unknown_fields_and_user() -> None:
    out = build("together", make_request(verbosity="low", user="bob", metadata={"a": "b"}))
    assert "verbosity" not in out.body
    assert "user" not in out.body
    assert "metadata" not in out.body
    assert {"verbosity", "user", "metadata"} <= set(out.ignored)
    assert "stream_options" not in out.body


def test_max_tokens_param_and_model_prefix() -> None:
    req = make_request(max_completion_tokens=77)
    fireworks = build("fireworks", req, model="gpt-oss-120b").body
    assert fireworks["model"] == "accounts/fireworks/models/gpt-oss-120b"
    assert fireworks["max_tokens"] == 77
    assert "max_completion_tokens" not in fireworks
    assert build("groq", req).body["max_completion_tokens"] == 77


def test_mistral_allow_list_renames_and_role_map() -> None:
    req = make_request(
        seed=7,
        logprobs=True,
        user="u",
        max_completion_tokens=20,
        temperature=1.9,
        messages=[{"role": "developer", "content": "be brief"}, *TOOL_HISTORY],
    )
    out = build("mistral", req)
    body = out.body
    assert body["random_seed"] == 7
    assert "seed" not in body
    assert "logprobs" not in body
    assert "stream_options" not in body
    assert body["messages"][0]["role"] == "system"
    assert body["temperature"] == 1.5
    assert "temperature:1.9->1.5" in out.adjustments
    assistant, tool = body["messages"][2], body["messages"][3]
    mapped = hashlib.sha256(b"call_abc").hexdigest()[:9]
    assert assistant["tool_calls"][0]["id"] == mapped
    assert tool["tool_call_id"] == mapped


def test_zai_tool_choice_stop_and_json_schema() -> None:
    req = make_request(
        tools=TOOLS,
        tool_choice="none",
        stop=["x"],
        response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": {}}},
    )
    out = build("zai", req)
    assert "tools" not in out.body
    assert "tool_choice" not in out.body
    assert out.body["response_format"] == {"type": "json_object"}
    assert "response_format:json_schema->json_object" in out.adjustments
    assert out.body["user_id"] == hashlib.sha256(b"key-a:").hexdigest()
    with pytest.raises(ProviderError) as forced:
        build("zai", make_request(tools=TOOLS, tool_choice="required"))
    assert forced.value.code == "capability_mismatch"
    assert "forced_tool_choice" in forced.value.violations
    with pytest.raises(ProviderError) as stops:
        build("zai", make_request(stop=["a", "b"]))
    assert stops.value.violations == ("stop",)


def test_ollama_tool_choice_temperature_and_extra_body() -> None:
    out = build(
        "ollama",
        make_request(tools=TOOLS, tool_choice="auto", parallel_tool_calls=True),
        defaults={"temperature": 0.4},
    )
    body = out.body
    assert "tool_choice" not in body
    assert "parallel_tool_calls" not in body
    assert body["tools"]
    assert body["temperature"] == 0.4
    assert body["keep_alive"] == "-1"
    assert build("ollama").body["temperature"] == 0.6


def test_strict_dropped_where_unsupported() -> None:
    out = build("deepinfra", make_request(tools=TOOLS))
    assert "strict" not in out.body["tools"][0]["function"]
    assert "tools.function.strict" in out.ignored
    assert build("openai", make_request(tools=TOOLS)).body["tools"][0]["function"]["strict"] is True


@pytest.mark.parametrize(
    ("provider", "effort", "expected"),
    [
        ("deepseek", "none", {"thinking": {"type": "disabled"}}),
        ("deepseek", "medium", {"reasoning_effort": "high"}),
        ("qwen", "none", {"enable_thinking": False}),
        ("qwen", "high", {"reasoning_effort": "xhigh"}),
        ("openrouter", "low", {"reasoning": {"effort": "low"}}),
        ("groq", "low", {"reasoning_effort": "low"}),
    ],
)
def test_reasoning_effort_mapping(provider: str, effort: str, expected: dict[str, Any]) -> None:
    body = build(provider, make_request(reasoning_effort=effort)).body
    for key, value in expected.items():
        assert body[key] == value
    if "thinking" in expected or "enable_thinking" in expected:
        assert "reasoning_effort" not in body


def test_openrouter_extra_body_merge() -> None:
    assert build("openrouter").body["provider"] == {"data_collection": "deny", "require_parameters": True}


def test_reasoning_history_injected_from_store() -> None:
    req = make_request(tools=TOOLS, messages=TOOL_HISTORY)
    key = history_keys(req)[0]
    assert key == "tc:call_abc"
    out = build("deepseek", req, history={key: "I should call f."})
    assert out.body["messages"][1]["reasoning_content"] == "I should call f."


def test_client_supplied_reasoning_kept_for_history_hosts_and_stripped_elsewhere() -> None:
    messages = [{**m, "reasoning_content": "kept"} if m["role"] == "assistant" else m for m in TOOL_HISTORY]
    req = make_request(tools=TOOLS, messages=messages)
    assert build("deepseek", req).body["messages"][1]["reasoning_content"] == "kept"
    assert "reasoning_content" not in build("together", req).body["messages"][1]


def test_missing_required_history_disables_thinking_or_rejects() -> None:
    req = make_request(tools=TOOLS, messages=TOOL_HISTORY, reasoning_effort="high")
    out = build("deepseek", req)
    assert out.body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in out.body
    assert "reasoning:disabled_missing_history" in out.adjustments
    strict = {"reasoning": {"history": "required_with_tools"}}
    with pytest.raises(ProviderError) as exc:
        build(strict, req)
    assert exc.value.violations == ("reasoning_history",)
    assert build(strict, make_request(messages=TOOL_HISTORY)).body


def test_history_key_for_text_turns() -> None:
    assert (
        history_key({"role": "assistant", "content": "hi"}) == "txt:" + hashlib.sha256(b"hi").hexdigest()[:32]
    )
    assert history_key({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}) == history_key(
        {"content": "hi"}
    )
    assert history_key({"role": "assistant"}) is None


def test_drop_reject_and_renames_with_dotted_paths() -> None:
    quirks = {"request": {"drop": ["seed"], "reject": ["logit_bias"], "renames": {"top_p": "sampling.top_p"}}}
    out = build(quirks, make_request(seed=1, top_p=0.5))
    assert "seed" not in out.body
    assert out.body["sampling"] == {"top_p": 0.5}
    with pytest.raises(ProviderError):
        build(quirks, make_request(logit_bias={"1": 2}))


def test_strip_message_fields_and_groq_name() -> None:
    req = make_request(messages=[{"role": "user", "content": "x", "name": "bob", "extra_content": {"a": 1}}])
    assert build("together", req).body["messages"][0] == {"role": "user", "content": "x", "name": "bob"}
    assert build("groq", req).body["messages"][0] == {"role": "user", "content": "x"}


def test_translation_is_deterministic() -> None:
    req = make_request(tools=TOOLS, user="u", temperature=0.1, messages=TOOL_HISTORY)
    assert build("openai", req).body == build("openai", req).body
