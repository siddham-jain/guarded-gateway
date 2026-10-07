from typing import Any

import pytest

from gg.cache.base import CacheKey
from gg.cache.keys import CacheKeyBuilder, default_alias_revision, number_signature, semantic_text
from gg.core.context import RequestContext
from gg.guardrails.base import POLICY_REF, PolicyRef
from tests.conftest import make_request
from tests.unit.cache.support import ctx_for

BUILDER = CacheKeyBuilder(lambda ctx: "rev1")


def key(**request: Any) -> CacheKey:
    return BUILDER.build(ctx_for(**request))


def test_key_is_deterministic_and_stable_across_processes() -> None:
    first = key(messages=[{"role": "user", "content": "hello"}])
    assert first == key(messages=[{"role": "user", "content": "hello"}])
    # a fixed digest: a change here means every deployed entry silently stops matching (bump v1)
    assert first.redis_key == (
        "gg:c:v1:k:test-key:21dc04243b5d50497f47a404c1e97348d6bbab7237f6517c57adfaeae994588b"
    )


def test_plain_string_equals_single_text_part() -> None:
    plain = key(messages=[{"role": "user", "content": "hello"}])
    parts = key(messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
    assert plain == parts


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"stream": True, "stream_options": {"include_usage": True}},
        {"user": "someone"},
        {"metadata": {"trace": "x"}},
        {"gg": {"cache": "no_store", "cache_ttl_s": 120, "semantic_cache": False}},
        {"n": 1},
        {"stop": []},
        {"logprobs": False},
        {"top_p": None},
    ],
)
def test_excluded_and_default_fields_do_not_change_the_key(extra: dict[str, Any]) -> None:
    assert key() == key(**extra)


@pytest.mark.parametrize(
    "extra",
    [
        {"top_p": 0.5},
        {"max_completion_tokens": 50},
        {"seed": 7},
        {"stop": ["\n"]},
        {"presence_penalty": 0.1},
        {"frequency_penalty": 0.1},
        {"response_format": {"type": "json_object"}},
        {"reasoning_effort": "low"},
        {"logprobs": True},
        {"gg": {"route_threshold": 0.4}},
        {"some_future_param": 1},
        {"model": "gg/strong"},
        {"messages": [{"role": "user", "content": "hello!"}]},
        {"messages": [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}]},
    ],
)
def test_each_param_changes_the_key(extra: dict[str, Any]) -> None:
    assert key().payload_sha != key(**extra).payload_sha


def test_temperature_is_part_of_the_key() -> None:
    assert key(temperature=0).payload_sha != key(temperature=0.5).payload_sha


def test_key_comes_from_the_scrubbed_request_not_the_original() -> None:
    ctx = ctx_for(messages=[{"role": "user", "content": "mail bob@corp.example"}])
    ctx.scrubbed = make_request(temperature=0, messages=[{"role": "user", "content": "mail [EMAIL_1]"}])
    other = ctx_for(messages=[{"role": "user", "content": "mail eve@corp.example"}])
    other.scrubbed = make_request(temperature=0, messages=[{"role": "user", "content": "mail [EMAIL_1]"}])
    assert BUILDER.build(ctx) == BUILDER.build(other)


def test_image_urls_are_hashed_not_embedded() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://img.example/cat.png"}}
    built = key(messages=[{"role": "user", "content": [{"type": "text", "text": "what is this"}, image]}])
    other = key(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "https://img.example/dog.png"}},
                ],
            }
        ]
    )
    assert built.payload_sha != other.payload_sha


def _with_scope(scope: str, key_id: str = "test-key") -> RequestContext:
    return ctx_for(key={"id": key_id, "cache": {"scope": scope}})


def test_scope_is_per_key_unless_global() -> None:
    a, b = _with_scope("key", "key-a"), _with_scope("key", "key-b")
    assert BUILDER.build(a).scope == "k:key-a"
    assert BUILDER.build(a).redis_key != BUILDER.build(b).redis_key
    assert BUILDER.build(a).tags.scope != BUILDER.build(b).tags.scope
    g1, g2 = _with_scope("global", "key-a"), _with_scope("global", "key-b")
    assert BUILDER.build(g1).scope == "g"
    assert BUILDER.build(g1).redis_key == BUILDER.build(g2).redis_key
    assert BUILDER.build(g1).redis_key != BUILDER.build(a).redis_key


def test_policy_hash_invalidates() -> None:
    ctx, other = ctx_for(), ctx_for()
    ctx.set(POLICY_REF, PolicyRef("default", "1", "aaa"))
    other.set(POLICY_REF, PolicyRef("default", "1", "bbb"))
    assert BUILDER.build(ctx).payload_sha != BUILDER.build(other).payload_sha
    assert BUILDER.build(ctx).tags.policy_sha != BUILDER.build(other).tags.policy_sha


def test_alias_revision_invalidates() -> None:
    ctx = ctx_for()
    assert CacheKeyBuilder(lambda c: "r1").build(ctx) != CacheKeyBuilder(lambda c: "r2").build(ctx)


def test_default_alias_revision_follows_config_and_key_routing() -> None:
    base = ctx_for()
    base.config_hash = "cfg1"
    changed_config = ctx_for()
    changed_config.config_hash = "cfg2"
    changed_alpha = ctx_for(key={"routing": {"threshold": 0.7}})
    changed_alpha.config_hash = "cfg1"
    revisions = {default_alias_revision(c) for c in (base, changed_config, changed_alpha)}
    assert len(revisions) == 3


def test_semantic_tags_split_system_params_and_numbers() -> None:
    plain = key(messages=[{"role": "user", "content": "what is 17 times 23"}])
    other_number = key(messages=[{"role": "user", "content": "what is 17 times 24"}])
    with_system = key(
        messages=[{"role": "system", "content": "terse"}, {"role": "user", "content": "what is 17 times 23"}]
    )
    other_params = key(messages=[{"role": "user", "content": "what is 17 times 23"}], max_completion_tokens=9)
    assert plain.tags.num_sig != other_number.tags.num_sig
    assert plain.tags.system_sha != with_system.tags.system_sha
    assert plain.tags.params_sha != other_params.tags.params_sha
    assert plain.tags.system_sha == other_number.tags.system_sha
    assert all(len(v) == 16 for v in plain.tags.as_dict().values())


def test_number_signature_ignores_order() -> None:
    assert number_signature("3 apples and 4 pears") == number_signature("4 pears and 3 apples")
    assert number_signature("3.5 kg") != number_signature("35 kg")


def test_semantic_text_normalises() -> None:
    fullwidth_hello = "\uff48\uff45\uff4c\uff4c\uff4f"
    request = make_request(messages=[{"role": "user", "content": f"  {fullwidth_hello}\n\n  world  "}])
    assert semantic_text(request) == "hello world"
