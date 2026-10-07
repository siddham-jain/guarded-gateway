from typing import Any

import pytest

from gg.api.authorize import authorize_request
from gg.core.errors import GGError, InvalidRequestError, NotFoundError, PermissionDeniedError
from gg.core.normalize import Normalization
from tests.conftest import make_key, make_request
from tests.unit.api.fakes import FakeCatalog

CATALOG = FakeCatalog()
OPEN = {"allowed_models": ["*"]}


def _authorize(key: dict[str, Any] | None = None, **request: Any) -> Any:
    return authorize_request(
        make_request(**request),
        make_key(**(OPEN if key is None else key)),
        CATALOG,
        consumed_extensions=frozenset({"fallback"}),
    )


@pytest.mark.parametrize(
    ("key", "model", "error", "code"),
    [
        ({}, "mock/echo", PermissionDeniedError, "model_not_allowed"),
        ({"allowed_models": ["mock/*"]}, "mock/nope", NotFoundError, "model_not_found"),
        ({"allowed_models": ["mock/*"]}, "mock/premium", PermissionDeniedError, "model_not_allowed"),
        (
            {"allowed_models": ["mock/*"], "flags": {"allow_free_tier_providers": False}},
            "mock/free",
            PermissionDeniedError,
            "model_not_allowed",
        ),
        (
            {"allowed_models": ["*"], "allowed_providers": ["mock"]},
            "other/big",
            PermissionDeniedError,
            "model_not_allowed",
        ),
    ],
)
def test_model_checks(key: dict[str, Any], model: str, error: type[GGError], code: str) -> None:
    with pytest.raises(error) as info:
        _authorize(key, model=model)
    assert info.value.code == code
    assert info.value.param == "model"


def test_allowed_models_and_flags_pass() -> None:
    assert _authorize({"allowed_models": ["mock/*"]}, model="mock/echo").request.model == "mock/echo"
    assert _authorize({"allowed_models": ["mock/*"], "flags": {"allow_premium": True}}, model="mock/premium")
    # aliases are filtered by provider at routing time, not here
    assert _authorize({"allowed_models": ["gg/*"], "allowed_providers": ["nobody"]}, model="gg/auto")


@pytest.mark.parametrize(
    ("limits", "fields", "param"),
    [
        ({"max_messages": 1}, {"messages": [{"role": "user", "content": "a"}] * 2}, "messages"),
        (
            {"max_tools": 1},
            {"tools": [{"type": "function", "function": {"name": f"f{i}"}} for i in range(2)]},
            "tools",
        ),
        ({"max_n": 1}, {"n": 2}, "n"),
        ({"max_completion_tokens": 10}, {"max_completion_tokens": 11}, "max_completion_tokens"),
    ],
)
def test_caps_reject(limits: dict[str, Any], fields: dict[str, Any], param: str) -> None:
    with pytest.raises(InvalidRequestError) as info:
        _authorize({**OPEN, "limits": limits}, model="mock/echo", **fields)
    assert info.value.param == param
    assert info.value.code == "invalid_value"


def test_caps_allow_unset_and_equal() -> None:
    _authorize(
        {**OPEN, "limits": {"max_completion_tokens": None}}, model="mock/echo", max_completion_tokens=10**6
    )
    _authorize({**OPEN, "limits": {"max_completion_tokens": 10}}, model="mock/echo", max_completion_tokens=10)


def test_policy_stripped_params() -> None:
    result = _authorize(OPEN, model="mock/echo", service_tier="priority", store=True, user="u1")
    assert result.ignored_params == {"service_tier", "store"}
    extra = result.request.model_extra or {}
    assert "service_tier" not in extra
    assert "store" not in extra
    assert result.request.user == "u1"


def test_store_false_and_allowed_flags_kept() -> None:
    result = _authorize(
        {**OPEN, "flags": {"allow_service_tier": True}}, model="mock/echo", service_tier="flex", store=False
    )
    assert result.ignored_params == frozenset()
    assert (result.request.model_extra or {})["service_tier"] == "flex"


def test_unconsumed_extensions_reported() -> None:
    result = _authorize(OPEN, model="mock/echo", gg={"fallback": False, "cache": "off", "tags": ["a"]})
    assert result.ignored_params == {"gg.cache", "gg.tags"}


def test_route_threshold_rules() -> None:
    allowed = {**OPEN, "routing": {"allow_request_threshold": True, "threshold_bounds": [0.2, 0.8]}}
    assert (
        "gg.route_threshold"
        in _authorize(allowed, model="gg/auto", gg={"route_threshold": 0.5}).ignored_params
    )
    with pytest.raises(PermissionDeniedError) as denied:
        _authorize(OPEN, model="gg/auto", gg={"route_threshold": 0.5})
    assert denied.value.code == "gg_override_not_allowed"
    assert denied.value.param == "gg.route_threshold"
    with pytest.raises(InvalidRequestError) as bounds:
        _authorize(allowed, model="gg/auto", gg={"route_threshold": 0.9})
    assert bounds.value.param == "gg.route_threshold"
    direct = _authorize(OPEN, model="mock/echo", gg={"route_threshold": 0.5})
    assert "gg.route_threshold" in direct.ignored_params


def test_guardrail_tightening_needs_permission() -> None:
    no = {**OPEN, "guardrails": {"allow_request_tightening": False}}
    with pytest.raises(PermissionDeniedError) as info:
        _authorize(no, model="mock/echo", gg={"guardrails": {"enforce_shadowed": True}})
    assert info.value.param == "gg.guardrails"
    assert (
        "gg.guardrails"
        in _authorize(OPEN, model="mock/echo", gg={"guardrails": {"enable": ["x"]}}).ignored_params
    )


def test_cache_ttl_bounded_by_key() -> None:
    with pytest.raises(InvalidRequestError) as info:
        _authorize({**OPEN, "cache": {"max_ttl_s": 60}}, model="mock/echo", gg={"cache_ttl_s": 61})
    assert info.value.param == "gg.cache_ttl_s"


def test_null_gg_is_absent() -> None:
    assert _authorize(OPEN, model="mock/echo", gg=None).ignored_params == frozenset()


def test_dropped_stream_options_reported() -> None:
    result = authorize_request(
        make_request(model="mock/echo"),
        make_key(**OPEN),
        CATALOG,
        normalizations=(
            Normalization("stream_options_dropped", "stream_options"),
            Normalization("max_tokens", "max_tokens"),
        ),
    )
    assert result.ignored_params == {"stream_options"}
