from typing import Any

import pytest

from gg.core.schema import ChatRequest
from gg.providers.catalog.capabilities import AdjustPolicy, CapabilityChecker, CheckResult
from gg.providers.usage import estimate_prompt_tokens
from tests.conftest import make_request
from tests.unit.providers.support import make_dep

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
IMAGE = [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]
DATA_IMAGE = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
REASONING = {"effort_levels": frozenset({"none", "low", "medium", "high"})}


def check(caps: dict[str, Any], policy: AdjustPolicy | None = None, **req: Any) -> CheckResult:
    return CapabilityChecker(estimate_prompt_tokens).check(make_request(**req), make_dep(**caps), policy)


@pytest.mark.parametrize(
    ("caps", "req", "reject"),
    [
        ({"tools": False}, {"tools": TOOLS}, "tools"),
        ({"forced_tool_choice": False}, {"tools": TOOLS, "tool_choice": "required"}, "forced_tool_choice"),
        (
            {"forced_tool_choice": False},
            {"tools": TOOLS, "tool_choice": {"type": "function", "function": {"name": "f"}}},
            "forced_tool_choice",
        ),
        (
            {"json_schema": False},
            {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
            "json_schema",
        ),
        ({"json_object": False}, {"response_format": {"type": "json_object"}}, "json_object"),
        ({"vision": False}, {"messages": [{"role": "user", "content": IMAGE}]}, "vision"),
        (
            {"vision": True, "image_url": "unsupported"},
            {"messages": [{"role": "user", "content": IMAGE}]},
            "image_url",
        ),
        ({"n": False}, {"n": 2}, "n"),
        ({"logprobs": False}, {"logprobs": True}, "logprobs"),
        ({"stop_max": 1}, {"stop": ["a", "b"]}, "stop"),
        (
            {"prefill": False},
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]},
            "prefill",
        ),
        ({"context": 10}, {"messages": [{"role": "user", "content": "word " * 200}]}, "context_length"),
        (
            {"audio": False},
            {
                "messages": [
                    {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "x"}}]}
                ]
            },
            "audio",
        ),
        (
            {"pdf": False},
            {"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_data": "x"}}]}]},
            "pdf",
        ),
    ],
)
def test_rejects(caps: dict[str, Any], req: dict[str, Any], reject: str) -> None:
    result = check(caps, **req)
    assert reject in result.rejects
    assert not result.ok


@pytest.mark.parametrize(
    ("caps", "req"),
    [
        ({"vision": True}, {"messages": [{"role": "user", "content": IMAGE}]}),
        (
            {"vision": True, "image_url": "unsupported"},
            {"messages": [{"role": "user", "content": DATA_IMAGE}]},
        ),
        ({"forced_tool_choice": False}, {"tools": TOOLS, "tool_choice": "auto"}),
        ({}, {"stop": ["a", "b", "c", "d"]}),
        (
            {"prefill": True},
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}]},
        ),
    ],
)
def test_accepts(caps: dict[str, Any], req: dict[str, Any]) -> None:
    assert check(caps, **req).ok


def test_forced_tool_choice_downgrade_policy() -> None:
    result = check(
        {"forced_tool_choice": False}, AdjustPolicy("downgrade"), tools=TOOLS, tool_choice="required"
    )
    assert result.ok
    assert result.patch["tool_choice"] == "auto"
    assert "tool_choice:forced->auto" in result.adjustments


@pytest.mark.parametrize(
    ("caps", "req", "effort", "adjustment"),
    [
        (
            {**REASONING, "effort_clamp": {"max": "high"}},
            {"reasoning_effort": "max"},
            "high",
            "reasoning_effort:max->high",
        ),
        (REASONING, {"reasoning_effort": "xhigh"}, "high", "reasoning_effort:xhigh->high"),
        (REASONING, {"reasoning_effort": "minimal"}, "low", "reasoning_effort:minimal->low"),
        ({**REASONING, "defaults": {"reasoning_effort": "low"}}, {}, "low", None),
        (
            {**REASONING, "tools_require_effort": "none"},
            {"reasoning_effort": "high", "tools": TOOLS},
            "none",
            "reasoning_effort:high->none",
        ),
    ],
)
def test_effort_clamp_and_defaults(
    caps: dict[str, Any], req: dict[str, Any], effort: str, adjustment: str | None
) -> None:
    result = check(caps, **req)
    assert result.patch["reasoning_effort"] == effort
    if adjustment:
        assert adjustment in result.adjustments


def test_effort_dropped_on_non_reasoning_model() -> None:
    result = check({}, reasoning_effort="high")
    assert result.patch["reasoning_effort"] is None
    assert "reasoning_effort" in result.ignored_params


@pytest.mark.parametrize(
    ("mode", "req", "stripped"),
    [
        ("none", {"temperature": 0.2, "top_p": 0.9}, {"temperature", "top_p"}),
        (
            "when_effort_none",
            {"temperature": 0.2, "logprobs": True, "reasoning_effort": "low"},
            {"temperature", "logprobs"},
        ),
        ("when_effort_none", {"temperature": 0.2, "reasoning_effort": "none"}, set()),
        ("temp_or_top_p", {"temperature": 0.2, "top_p": 0.9}, {"top_p"}),
        ("advisory", {"temperature": 0.5}, {"temperature"}),
        ("advisory", {"temperature": 1.2}, set()),
        ("full", {"temperature": 0.2, "top_p": 0.9}, set()),
    ],
)
def test_sampling_rules(mode: str, req: dict[str, Any], stripped: set[str]) -> None:
    result = check({**REASONING, "sampling_params": mode}, **req)
    assert set(result.ignored_params) == stripped
    assert all(result.patch[name] is None for name in stripped)


def test_temperature_clamp_and_max_tokens() -> None:
    result = check({"temperature_max": 1.0, "max_output": 100}, temperature=1.7, max_completion_tokens=500)
    assert result.patch["temperature"] == 1.0
    assert result.patch["max_completion_tokens"] == 100
    assert "max_completion_tokens:500->100" in result.adjustments
    defaulted = check({"defaults": {"max_completion_tokens": 64}})
    assert defaulted.patch["max_completion_tokens"] == 64
    assert not defaulted.adjustments
    floor = check({"min_max_tokens": 1024, "max_output": 4096}, max_completion_tokens=10)
    assert floor.patch["max_completion_tokens"] == 1024


def test_apply_never_mutates_and_compatible_filters() -> None:
    checker = CapabilityChecker()
    request = make_request(temperature=0.3)
    result = checker.check(request, make_dep(sampling_params="none"))
    applied = checker.apply(request, result)
    assert request.temperature == 0.3
    assert applied.temperature is None
    assert isinstance(applied, ChatRequest)
    deps = [make_dep("a", tools=False), make_dep("b")]
    assert [d.id for d in checker.compatible(make_request(tools=TOOLS), deps)] == ["b/test-model"]
