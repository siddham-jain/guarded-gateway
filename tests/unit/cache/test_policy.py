from typing import Any

import pytest

from gg.cache.policy import PII_UNREDACTED, decide, threshold_for, ttl_for
from gg.core.cache_types import CacheState
from gg.core.context import RequestContext
from gg.guardrails.vault import GuardVault
from tests.unit.cache.support import config, ctx_for

CFG = config()
ONE_TURN = [{"role": "user", "content": "what is the capital of france"}]


def _decide(ctx: RequestContext, *, backend_up: bool = True, index_up: bool = True) -> Any:
    return decide(ctx, CFG, backend_up=backend_up, index_up=index_up)


@pytest.mark.parametrize(
    ("request_fields", "key_fields", "reason"),
    [
        ({}, {}, None),
        ({"temperature": None}, {}, "sampled"),
        ({"temperature": 0.7}, {}, "sampled"),
        ({"temperature": 0.7, "seed": 3}, {}, None),
        ({"temperature": None, "seed": 3}, {}, None),
        ({"temperature": 0.7}, {"cache": {"allow_sampled": True}}, None),
        ({"n": 2}, {"limits": {"max_n": 4}}, "n_gt_1"),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, {}, "tools"),
        ({"tool_choice": "none"}, {}, "tools"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}}],
                    }
                ]
            },
            {},
            "unsupported_content",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "f1"}}]}]},
            {},
            "unsupported_content",
        ),
        ({"gg": {"cache": "off"}}, {}, "client_off"),
        ({}, {"cache": {"scope": "off"}}, "disabled"),
    ],
)
def test_cacheability_matrix(
    request_fields: dict[str, Any], key_fields: dict[str, Any], reason: str | None
) -> None:
    decision = _decide(ctx_for(key=key_fields or None, **request_fields))
    assert decision.reason == reason
    assert decision.bypass is (reason is not None)


def test_exact_layer_switch() -> None:
    off = config(enabled=False)
    assert decide(ctx_for(), off, backend_up=True, index_up=True).reason == "disabled"


def test_backend_down_bypasses() -> None:
    assert _decide(ctx_for(), backend_up=False).reason == "backend_down"


def test_unredacted_upstream_pii_bypasses_both_layers() -> None:
    ctx = ctx_for()
    ctx.cache = CacheState(bypass_reason="unredacted_upstream", store=False)
    decision = _decide(ctx)
    assert decision.bypass
    assert decision.reason == PII_UNREDACTED
    assert not decision.semantic


@pytest.mark.parametrize(
    ("mode", "lookup", "store"),
    [("default", True, True), ("refresh", False, True), ("no_store", True, False)],
)
def test_client_cache_modes(mode: str, lookup: bool, store: bool) -> None:
    decision = _decide(ctx_for(gg={"cache": mode}))
    assert (decision.lookup, decision.store) == (lookup, store)
    assert decision.reason == (None if lookup else "client_refresh")


def test_ttl_defaults_to_key_and_clamps() -> None:
    assert ttl_for(ctx_for(), CFG) == 86_400
    assert ttl_for(ctx_for(gg={"cache_ttl_s": 5}), CFG) == 60
    assert ttl_for(ctx_for(gg={"cache_ttl_s": 600}), CFG) == 600
    capped = ctx_for(key={"cache": {"max_ttl_s": 300}}, gg={"cache_ttl_s": 600})
    assert ttl_for(capped, CFG) == 300
    assert ttl_for(ctx_for(key={"cache": {"default_ttl_s": 900_000}}), CFG) == 604_800


def test_request_threshold_only_tightens() -> None:
    assert threshold_for(ctx_for(), CFG) == pytest.approx(0.08)
    assert threshold_for(ctx_for(gg={"cache_threshold": 0.02}), CFG) == pytest.approx(0.02)
    assert threshold_for(ctx_for(gg={"cache_threshold": 0.5}), CFG) == pytest.approx(0.08)


SEMANTIC_KEY = {"cache": {"semantic": True}}


@pytest.mark.parametrize(
    ("request_fields", "key_fields", "reason"),
    [
        ({"messages": ONE_TURN}, SEMANTIC_KEY, None),
        ({"messages": [{"role": "system", "content": "be terse"}, *ONE_TURN]}, SEMANTIC_KEY, None),
        ({"messages": ONE_TURN}, {}, "sem_disabled"),
        ({"messages": ONE_TURN, "gg": {"semantic_cache": False}}, SEMANTIC_KEY, "sem_disabled"),
        (
            {"messages": [*ONE_TURN, {"role": "assistant", "content": "Paris"}, *ONE_TURN]},
            SEMANTIC_KEY,
            "multi_turn",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is in this picture"},
                            {"type": "image_url", "image_url": {"url": "https://x.example/a.png"}},
                        ],
                    }
                ]
            },
            SEMANTIC_KEY,
            "non_text",
        ),
        ({"messages": [{"role": "user", "content": "hi"}]}, SEMANTIC_KEY, "too_short"),
        ({"messages": [{"role": "user", "content": "x" * 1300}]}, SEMANTIC_KEY, "too_long"),
    ],
)
def test_semantic_eligibility(
    request_fields: dict[str, Any], key_fields: dict[str, Any], reason: str | None
) -> None:
    decision = _decide(ctx_for(key=key_fields or None, **request_fields))
    assert not decision.bypass
    assert decision.semantic_reason == reason
    assert decision.semantic is (reason is None)


def test_semantic_needs_an_empty_vault() -> None:
    ctx = ctx_for(key=SEMANTIC_KEY, messages=ONE_TURN)
    vault = GuardVault()
    vault.add("EMAIL", "a@b.example")
    ctx.vault = vault
    assert _decide(ctx).semantic_reason == "has_placeholders"


def test_semantic_needs_the_index() -> None:
    ctx = ctx_for(key=SEMANTIC_KEY, messages=ONE_TURN)
    assert _decide(ctx, index_up=False).semantic_reason == "sem_unavailable"


def test_semantic_layer_switch() -> None:
    cfg = config().model_copy(update={"semantic": config().semantic.model_copy(update={"enabled": False})})
    ctx = ctx_for(key=SEMANTIC_KEY, messages=ONE_TURN)
    assert decide(ctx, cfg, backend_up=True, index_up=True).semantic_reason == "sem_disabled"
