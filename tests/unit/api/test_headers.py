import re

from gg.api.headers import build_response_headers, stage_headers
from gg.core.clock import FakeClock
from gg.core.routing_types import PlanEntry, RouteDecision, RoutePlan
from gg.core.usage import AttemptRecord
from tests.conftest import make_ctx
from tests.unit.api.fakes import ECHO, OTHER


def test_minimal_headers(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    headers = build_response_headers(ctx, streaming=False)
    assert headers == {}
    ctx.timings.mark("received")
    assert build_response_headers(ctx, streaming=False) == {"server-timing": "gw;dur=0.00"}


def test_served_headers(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ctx.config_hash = "abcdef0123456789"
    ctx.served_by = OTHER
    ctx.route = RouteDecision(
        alias="gg/auto",
        tier="weak",
        reason="threshold",
        plan=RoutePlan(alias="gg/auto", entries=(PlanEntry(ECHO), PlanEntry(OTHER))),
    )
    ctx.attempts = [
        AttemptRecord("mock/echo", "mock", 0, 1, "fallback", upstream_request_id="up-1"),
        AttemptRecord("other/big", "other", 1, 1, "ok", upstream_request_id="up-2"),
    ]
    ctx.ignored_params = {"store", "logit_bias"}
    headers = build_response_headers(ctx, streaming=False)
    assert headers["x-gg-config"] == "abcdef012345"
    assert headers["x-gg-provider"] == "other"
    assert headers["x-gg-model"] == "other/big"
    assert headers["x-gg-fallback"] == "true"
    assert headers["x-gg-attempts"] == "2"
    assert headers["x-gg-upstream-request-id"] == "up-2"
    assert headers["x-gg-ignored-params"] == "logit_bias,store"


def test_early_commit_omits_served_headers(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ctx.served_by = ECHO
    headers = build_response_headers(ctx, streaming=True, early_commit=True)
    assert "x-gg-provider" not in headers
    assert headers["content-type"] == "text/event-stream; charset=utf-8"
    assert headers["cache-control"] == "no-cache"
    assert headers["x-accel-buffering"] == "no"


def test_stage_headers_filtered(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ctx.response_headers.update(
        {
            "X-GG-Cache": "HIT",
            "x-ratelimit-remaining-requests": "5",
            "retry-after": "3",
            "set-cookie": "a=b",
            "x-gg-bad": "café",
            "x-gg-long": "x" * 600,
            "x-gg-newline": "a\nb",
        }
    )
    assert stage_headers(ctx) == {
        "x-gg-cache": "HIT",
        "x-ratelimit-remaining-requests": "5",
        "retry-after": "3",
    }


def test_server_timing_format(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    ctx.timings.mark("received")
    ctx.timings.record("auth", 0.0004)
    ctx.timings.record("terminal", 0.5)
    clock.advance(0.6)
    timing = build_response_headers(ctx, streaming=False)["server-timing"]
    assert re.fullmatch(r"auth;dur=0\.40, terminal;dur=500\.00, gw;dur=100\.00", timing)
