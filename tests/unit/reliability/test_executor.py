from dataclasses import replace
from datetime import timedelta

import pytest

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import (
    ContextLengthError,
    InternalError,
    InvalidRequestError,
    RateLimitedError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gg.core.usage import UsageRecord
from gg.reliability.breaker import Cooldown
from tests.conftest import make_ctx
from tests.unit.reliability.fakes import (
    USAGE,
    FakeAdapter,
    Harness,
    Step,
    context_for,
    deployment,
    drain,
    fail,
    ok,
    overloaded,
    stall,
)

A = deployment("a/model-a")
B = deployment("b/model-b")


def harness(clock: FakeClock, a: list[Step] | None = None, b: list[Step] | None = None) -> Harness:
    return Harness(clock, {"a": FakeAdapter("a", a or []), "b": FakeAdapter("b", b or [])})


def outcomes(ctx: RequestContext) -> list[str]:
    return [r.outcome for r in ctx.attempts]


async def test_retry_then_success(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("retryable", status=500), ok()])
    ctx = context_for(clock, [A, B])

    result = await h.executor(ctx)

    assert result.response is not None
    assert result.response.choices[0].message.content == "hello world"
    assert outcomes(ctx) == ["retry", "ok"]
    assert len(h.sleep.calls) == 1
    assert 0 <= h.sleep.calls[0] <= 0.25
    assert h.adapters["b"].calls == 0
    assert ctx.served_by == A
    assert ctx.response_headers == {"x-gg-provider": "a", "x-gg-model": "a/model-a", "x-gg-attempts": "2"}


async def test_fallback_across_providers(clock: FakeClock) -> None:
    h = harness(clock, a=[overloaded()])
    ctx = context_for(clock, [A, B])

    result = await h.executor(ctx)

    assert result.response is not None
    assert ctx.served_by == B
    assert outcomes(ctx) == ["fallback", "ok"]
    assert h.sleep.calls == []
    assert ctx.response_headers["x-gg-provider"] == "b"
    assert ctx.response_headers["x-gg-attempts"] == "2"


async def test_client_error_stops_immediately(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("client", status=400, code="invalid_value")])
    ctx = context_for(clock, [A, B])

    with pytest.raises(InvalidRequestError) as exc:
        await h.executor(ctx)

    assert exc.value.message.startswith("fake: ")
    assert h.adapters["b"].calls == 0
    assert outcomes(ctx) == ["fail"]
    assert ctx.response_headers["x-gg-attempts"] == "1"


async def test_content_filter_does_not_fall_back(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("content_filter", status=400)])
    with pytest.raises(InvalidRequestError) as exc:
        await h.executor(context_for(clock, [A, B]))
    assert exc.value.code == "content_filter"
    assert h.adapters["b"].calls == 0


async def test_all_fail_maps_to_one_error(clock: FakeClock) -> None:
    h = harness(clock, a=[overloaded()], b=[fail("retryable", status=500), fail("retryable", status=500)])
    ctx = context_for(clock, [A, B])

    with pytest.raises(ServiceUnavailableError) as exc:
        await h.executor(ctx)

    assert exc.value.code == "upstream_overloaded"
    assert exc.value.response_headers()["x-should-retry"] == "true"
    assert outcomes(ctx) == ["fallback", "retry", "fallback"]


async def test_stream_error_before_first_chunk_is_transparent(clock: FakeClock) -> None:
    h = harness(clock, a=[overloaded()])
    ctx = context_for(clock, [A, B], stream=True)

    result = await h.executor(ctx)

    assert result.stream is not None
    assert await drain(result.stream) == "hello world"
    assert ctx.served_by == B


async def test_stream_error_after_first_chunk_is_raised_in_stream(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("fallback", status=529, code="overloaded", after=("partial",))])
    ctx = context_for(clock, [A, B], stream=True)

    result = await h.executor(ctx)
    assert result.stream is not None
    first = await anext(result.stream)
    assert first.choices[0].delta.content == "partial"
    with pytest.raises(UpstreamError) as exc:
        await anext(result.stream)

    assert exc.value.code == "upstream_stream_error"
    assert "output is incomplete" in exc.value.message
    assert h.adapters["a"].calls == 1
    assert h.adapters["b"].calls == 0
    assert h.adapters["a"].closed == 1
    assert h.breakers.get(A).snapshot().consecutive_failures == 1
    assert "upstream_end" in ctx.timings.marks


async def test_non_stream_mid_generation_failure_falls_back(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("fallback", status=529, code="overloaded", after=("partial",))])
    ctx = context_for(clock, [A, B])

    result = await h.executor(ctx)

    assert result.response is not None
    assert result.response.choices[0].message.content == "hello world"
    assert ctx.served_by == B


async def test_ttft_timeout_falls_back(clock: FakeClock) -> None:
    slow = deployment("a/model-a", ttft_s=0.02)
    h = harness(clock, a=[stall(5.0)])
    ctx = context_for(clock, [slow, B], stream=True)

    result = await h.executor(ctx)

    assert result.stream is not None
    assert await drain(result.stream) == "hello world"
    assert ctx.attempts[0].outcome == "fallback"
    assert ctx.attempts[0].status == 504
    assert h.adapters["a"].cancelled == 1
    assert h.adapters["a"].closed == 1


async def test_inter_chunk_timeout_after_commit(clock: FakeClock) -> None:
    slow = deployment("a/model-a", inter_chunk_s=0.02)
    h = harness(clock, a=[Step(chunks=("partial",), stall_after_s=5.0)])
    ctx = context_for(clock, [slow, B], stream=True)

    result = await h.executor(ctx)
    assert result.stream is not None
    await anext(result.stream)
    with pytest.raises(UpstreamTimeoutError) as exc:
        await anext(result.stream)

    assert exc.value.code == "upstream_timeout"
    assert h.adapters["b"].calls == 0
    assert h.adapters["a"].cancelled == 1


async def test_inter_chunk_timeout_before_commit_retries_non_stream(clock: FakeClock) -> None:
    slow = deployment("a/model-a", inter_chunk_s=0.02)
    h = harness(clock, a=[Step(chunks=("partial",), stall_after_s=5.0), ok()])
    ctx = context_for(clock, [slow, B])

    result = await h.executor(ctx)

    assert result.response is not None
    assert result.response.choices[0].message.content == "hello world"
    assert outcomes(ctx) == ["retry", "ok"]


async def test_empty_stream_is_retryable(clock: FakeClock) -> None:
    h = harness(clock, a=[Step(chunks=(), finish=False, usage=None), ok()])
    ctx = context_for(clock, [A, B])

    await h.executor(ctx)

    assert outcomes(ctx) == ["retry", "ok"]
    assert ctx.served_by == A


async def test_breaker_opens_and_skips_deployment(clock: FakeClock) -> None:
    h = harness(clock)
    h.adapters["a"].next_step = overloaded

    for _ in range(3):
        await h.executor(context_for(clock, [A, B]))
    ctx = context_for(clock, [A, B])
    await h.executor(ctx)

    assert h.adapters["a"].calls == 3
    assert h.breakers.get(A).state == "open"
    assert outcomes(ctx) == ["ok"]
    assert ctx.served_by == B

    clock.advance(30)
    h.adapters["a"].next_step = ok
    ctx = context_for(clock, [A, B])
    await h.executor(ctx)
    assert ctx.served_by == A
    assert h.breakers.get(A).state == "closed"


async def test_every_entry_open_returns_503_with_retry_after(clock: FakeClock) -> None:
    h = harness(clock)
    h.breakers.force_open(A, Cooldown(20, "trip"))
    h.breakers.force_open(B, Cooldown(45, "trip"))

    with pytest.raises(ServiceUnavailableError) as exc:
        await h.executor(context_for(clock, [A, B]))

    assert exc.value.code == "no_healthy_deployment"
    assert exc.value.retry_after_s == 20
    assert exc.value.response_headers()["x-should-retry"] == "false"
    assert h.adapters["a"].calls == h.adapters["b"].calls == 0


async def test_quota_day_opens_breaker_until_reset(clock: FakeClock) -> None:
    reset = clock.now() + timedelta(hours=2)
    h = harness(clock, a=[fail("quota_day", status=429, code="per_day", quota_reset_at=reset)])

    ctx = context_for(clock, [A, B])
    await h.executor(ctx)

    assert ctx.served_by == B
    assert h.breakers.get(A).retry_in() == pytest.approx(7200)
    await h.executor(context_for(clock, [A, B]))
    assert h.adapters["a"].calls == 1

    clock.advance(7200)
    ctx = context_for(clock, [A, B])
    await h.executor(ctx)
    assert ctx.served_by == A


async def test_auth_falls_back_and_opens_provider_breakers(clock: FakeClock) -> None:
    a2 = deployment("a/model-a2")
    h = harness(clock, a=[fail("auth", status=401, scope="provider")])

    ctx = context_for(clock, [A, B])
    await h.executor(ctx)

    assert ctx.served_by == B
    assert h.breakers.get(A).retry_in() == pytest.approx(600)
    assert h.breakers.get(a2).state == "open"


async def test_retry_after_within_wait_retries_same_deployment(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("quota_minute", status=429, retry_after_s=1.0), ok()])
    ctx = context_for(clock, [A, B])

    await h.executor(ctx)

    assert ctx.served_by == A
    assert 1.0 <= h.sleep.calls[0] <= 1.25


async def test_retry_after_larger_than_deadline_falls_back_and_cools_down(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("quota_minute", status=429, retry_after_s=60.0)])
    ctx = context_for(clock, [A, B])

    await h.executor(ctx)

    assert ctx.served_by == B
    assert h.sleep.calls == []
    assert h.breakers.get(A).retry_in() == pytest.approx(60)


async def test_retry_after_past_remaining_deadline_falls_back(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("quota_minute", status=429, retry_after_s=2.0)])
    ctx = context_for(clock, [A, B])
    ctx.deadline = ctx.deadline.child(3.0)

    await h.executor(ctx)

    assert ctx.served_by == B
    assert h.sleep.calls == []
    assert h.breakers.get(A).state == "closed"


async def test_rate_limited_everywhere_returns_429(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("quota_minute", status=429, retry_after_s=60.0)])
    with pytest.raises(RateLimitedError) as exc:
        await h.executor(context_for(clock, [A]))
    assert exc.value.code == "upstream_rate_limited"
    assert exc.value.retry_after_s == 60


@pytest.mark.parametrize("via", ["request", "plan"])
async def test_fallback_disabled_pins_first_entry(clock: FakeClock, via: str) -> None:
    h = harness(clock, a=[fail("retryable", status=500), overloaded()])
    if via == "request":
        ctx = context_for(clock, [A, B], gg={"fallback": False})
    else:
        ctx = context_for(clock, [A, B], allow_fallback=False)

    with pytest.raises(ServiceUnavailableError):
        await h.executor(ctx)

    assert h.adapters["a"].calls == 2
    assert h.adapters["b"].calls == 0


async def test_context_length_falls_back_to_larger_context(clock: FakeClock) -> None:
    small = deployment("a/small", context=8_000)
    smaller = deployment("b/smaller", context=4_000)
    large = deployment("c/large", context=200_000)
    adapters = {
        "a": FakeAdapter("a", [fail("fallback", status=400, code="context_length")]),
        "b": FakeAdapter("b"),
        "c": FakeAdapter("c"),
    }
    h = Harness(clock, adapters)
    ctx = context_for(clock, [small, smaller, large])

    await h.executor(ctx)

    assert ctx.served_by == large
    assert adapters["b"].calls == 0
    assert h.breakers.get(small).state == "closed"


async def test_context_length_without_larger_entry_is_400(clock: FakeClock) -> None:
    small = deployment("a/small", context=8_000)
    smaller = deployment("b/smaller", context=4_000)
    h = harness(clock, a=[fail("fallback", status=400, code="context_length")])

    with pytest.raises(ContextLengthError):
        await h.executor(context_for(clock, [small, smaller]))


async def test_capability_mismatch_does_not_trip_breaker(clock: FakeClock) -> None:
    h = harness(clock)
    h.adapters["a"].next_step = lambda: fail("fallback", status=0, code="capability_mismatch")

    for _ in range(5):
        ctx = context_for(clock, [A, B])
        await h.executor(ctx)
        assert ctx.served_by == B

    assert h.breakers.get(A).state == "closed"
    assert h.adapters["a"].calls == 5


async def test_tier_change_entries_skipped_unless_allowed(clock: FakeClock) -> None:
    h = harness(clock, a=[overloaded()])
    with pytest.raises(ServiceUnavailableError):
        await h.executor(context_for(clock, [A, B], tier_change=frozenset({B.id})))
    assert h.adapters["b"].calls == 0


async def test_max_hops_bounds_deployments_tried(clock: FakeClock) -> None:
    deps = [deployment(f"p{i}/m") for i in range(6)]
    adapters = {f"p{i}": FakeAdapter(f"p{i}", next_step=overloaded) for i in range(6)}
    h = Harness(clock, adapters)

    with pytest.raises(ServiceUnavailableError):
        await h.executor(context_for(clock, deps))

    assert [adapters[f"p{i}"].calls for i in range(6)] == [1, 1, 1, 1, 0, 0]


async def test_missing_adapter_is_skipped(clock: FakeClock) -> None:
    h = harness(clock)
    ctx = context_for(clock, [deployment("zz/unknown"), B])
    await h.executor(ctx)
    assert ctx.served_by == B


async def test_deadline_too_short_is_504(clock: FakeClock) -> None:
    h = harness(clock)
    ctx = context_for(clock, [A, B])
    ctx.deadline = ctx.deadline.child(1.0)

    with pytest.raises(UpstreamTimeoutError):
        await h.executor(ctx)
    assert h.adapters["a"].calls == 0


async def test_missing_route_is_internal_error(clock: FakeClock) -> None:
    h = harness(clock)
    with pytest.raises(InternalError):
        await h.executor(make_ctx(clock))


async def test_usage_and_timings_captured_non_stream(clock: FakeClock) -> None:
    h = harness(clock)
    ctx = context_for(clock, [A])

    result = await h.executor(ctx)

    assert result.response is not None
    assert result.response.usage == USAGE
    assert ctx.usage == UsageRecord(
        provider="a",
        deployment_id="a/model-a",
        upstream_model="model-a",
        input_tokens=3,
        output_tokens=2,
        raw=USAGE.model_dump(mode="json"),
    )
    assert {"upstream_start", "upstream_first_token", "upstream_end"} <= ctx.timings.marks.keys()


async def test_usage_captured_from_stream(clock: FakeClock) -> None:
    h = harness(clock)
    ctx = context_for(clock, [A], stream=True)

    result = await h.executor(ctx)
    assert ctx.usage is None
    assert result.stream is not None
    await drain(result.stream)

    assert ctx.usage is not None
    assert (ctx.usage.input_tokens, ctx.usage.output_tokens) == (3, 2)


async def test_stream_aclose_reaches_adapter(clock: FakeClock) -> None:
    h = harness(clock)
    ctx = context_for(clock, [A], stream=True)

    result = await h.executor(ctx)
    assert result.stream is not None
    await anext(result.stream)
    await result.stream.aclose()

    assert h.adapters["a"].closed == 1
    assert "upstream_end" in ctx.timings.marks


async def test_stream_aclose_before_iteration_reaches_adapter(clock: FakeClock) -> None:
    h = harness(clock)
    result = await h.executor(context_for(clock, [A], stream=True))
    assert result.stream is not None
    await result.stream.aclose()
    assert h.adapters["a"].closed == 1


async def test_every_attempt_translates_canonical_request(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("retryable", status=500), ok()])
    ctx = context_for(clock, [A])
    await h.executor(ctx)
    assert h.adapters["a"].requests == [ctx.request, ctx.request]


async def test_retry_errors_carry_kinds(clock: FakeClock) -> None:
    h = harness(clock, a=[fail("retryable", status=503), fail("retryable", status=503)])
    ctx = context_for(clock, [A, B])
    await h.executor(ctx)
    assert [(r.error_kind, r.status) for r in ctx.attempts] == [
        ("retryable", 503),
        ("retryable", 503),
        (None, None),
    ]


async def test_plan_entry_overrides_reach_the_adapter_without_touching_ctx(clock: FakeClock) -> None:
    h = harness(clock, a=[overloaded()], b=[ok()])
    ctx = context_for(clock, [A, B])
    assert ctx.route is not None
    entries = tuple(replace(e, overrides={"reasoning_effort": "low"}) for e in ctx.route.plan.entries)
    ctx.route = replace(ctx.route, plan=replace(ctx.route.plan, entries=entries))

    await h.executor(ctx)

    assert [r.reasoning_effort for r in h.adapters["a"].requests + h.adapters["b"].requests] == ["low", "low"]
    assert ctx.request.reasoning_effort is None
