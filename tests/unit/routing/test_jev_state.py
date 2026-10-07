from typing import Any

import pytest

from gg.core.jsonutil import canonical_json
from gg.routing.config import JevStateConfig
from gg.routing.scorers.jev.state import StateBuilder
from tests.unit.routing.support import all_fixtures, fixture, routing_request, user

BUILDER = StateBuilder(JevStateConfig())


def state(*messages: dict[str, Any], tools: bool = False) -> dict[str, Any]:
    built = BUILDER.build(routing_request(*messages, tools=tools))
    assert built.state is not None
    return built.state


@pytest.mark.parametrize("name", all_fixtures())
def test_matches_the_state_sent_in_spike_s2(name: str) -> None:
    recorded = fixture(name)["state"]
    turns: list[str] = recorded.get("recent_user_turns", [])
    messages = [user(t) for t in reversed(turns)]
    if turns:
        messages.append({"role": "assistant", "content": "noted"})
    messages.append(user(recorded["request"]))
    built = state(*messages)
    built.pop("last_assistant_message", None)
    assert built == recorded


def test_single_greeting() -> None:
    assert state(user("hi   there!\n")) == {
        "request": "hi there!",
        "context": {"conversation_depth": "new", "tools_offered": "none"},
    }


@pytest.mark.parametrize(("length", "truncated"), [(3999, False), (4000, False), (4001, True)])
def test_truncation_boundary(length: int, truncated: bool) -> None:
    text = "a" * (length - 1) + "Z"
    request = state(user(text))["request"]
    if not truncated:
        assert request == text
        return
    assert request.startswith("a" * 3000 + " … [truncated 101 chars] … ")
    assert request.endswith("Z")
    assert len(request) < length


def test_recent_turns_newest_first_and_capped() -> None:
    built = state(user("one"), user("two"), user("three " + "x" * 700), user("now"))
    assert built["recent_user_turns"] == [("three " + "x" * 700)[:600], "two"]
    assert built["context"]["conversation_depth"] == "short"


@pytest.mark.parametrize(("words", "included"), [(29, True), (30, False)])
def test_followup_includes_last_assistant(words: int, included: bool) -> None:
    built = state(user("plan it"), {"role": "assistant", "content": "Step 1"}, user(" ".join(["ok"] * words)))
    assert ("last_assistant_message" in built) is included


def test_tool_only_assistant_is_summarised_and_tool_results_never_sent() -> None:
    built = state(
        user("check the weather"),
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "geo", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "SECRET TOOL OUTPUT"},
        user("and tomorrow?"),
        tools=True,
    )
    assert built["last_assistant_message"] == "(assistant called tools: weather, geo)"
    assert built["context"]["tools_offered"] == "some"
    assert "SECRET TOOL OUTPUT" not in canonical_json(built).decode()


def test_images_and_system_excerpt_and_placeholders() -> None:
    built = state(
        {"role": "system", "content": "You help [EMAIL_1]. " + "y" * 600},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    )
    assert built["request"] == "what is this? [image attached]"
    excerpt = built["context"]["system_prompt_excerpt"]
    assert excerpt.startswith("You help [EMAIL_1].")
    assert len(excerpt) == 500


def test_system_excerpt_can_be_disabled() -> None:
    builder = StateBuilder(JevStateConfig(include_system_excerpt=False))
    built = builder.build(routing_request({"role": "system", "content": "secret ops"}, user("hi")))
    assert built.state is not None
    assert "system_prompt_excerpt" not in built.state["context"]


def test_long_conversation_bucket() -> None:
    assert state(*[user(f"t{i}") for i in range(6)])["context"]["conversation_depth"] == "long"


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "only system"}],
        [user("   ")],
    ],
)
def test_unscorable(messages: list[dict[str, Any]]) -> None:
    assert BUILDER.build(routing_request(*messages)).unscorable


def test_cache_key_is_stable_and_versioned() -> None:
    a = BUILDER.build(routing_request(user("hello")))
    b = BUILDER.build(routing_request(user("hello ")))
    assert a.cache_key("v1") == b.cache_key("v1")
    assert a.cache_key("v1") != a.cache_key("v2")
