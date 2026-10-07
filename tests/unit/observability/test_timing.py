import pytest

from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.usage import AttemptRecord, UsageRecord
from gg.observability.timing import compute_timings, final_attempt
from tests.conftest import make_ctx, make_request


def marked(clock: FakeClock, *, stream: bool, marks: dict[str, float]) -> RequestContext:
    ctx = make_ctx(clock, make_request(stream=stream))
    base = ctx.received_at
    ctx.timings.mark("received", base)
    for name, offset in marks.items():
        ctx.timings.mark(name, base + offset)
    return ctx


def test_non_stream_overhead_from_marks(clock: FakeClock) -> None:
    ctx = marked(
        clock,
        stream=False,
        marks={
            "upstream_start": 0.010,
            "upstream_first_token": 0.110,
            "upstream_end": 0.210,
            "completed": 0.215,
        },
    )
    t = compute_timings(ctx, now=999.0)
    assert t.total == pytest.approx(0.215)
    assert t.pre_upstream == pytest.approx(0.010)
    assert t.upstream == pytest.approx(0.200)
    assert t.upstream_ttft == pytest.approx(0.100)
    assert t.post == pytest.approx(0.005)
    assert t.overhead == pytest.approx(0.015)
    assert t.ttft is None
    assert t.stream_tail is None
    assert t.overhead_phases() == pytest.approx({"pre_upstream": 0.010, "post": 0.005, "total": 0.015})


def test_stream_ttft_added_and_tail(clock: FakeClock) -> None:
    ctx = marked(
        clock,
        stream=True,
        marks={
            "upstream_start": 0.020,
            "upstream_first_token": 0.320,
            "client_first_byte": 0.325,
            "upstream_end": 1.000,
            "completed": 1.002,
        },
    )
    ctx.usage = UsageRecord(
        provider="openai", deployment_id="d", upstream_model="m", input_tokens=10, output_tokens=69
    )
    t = compute_timings(ctx, now=999.0)
    assert t.ttft == pytest.approx(0.325)
    assert t.upstream_ttft == pytest.approx(0.300)
    assert t.ttft_added == pytest.approx(0.025)
    assert t.stream_tail == pytest.approx(0.002)
    assert t.overhead == pytest.approx(0.027)
    assert t.tpot == pytest.approx(0.680 / 68)
    assert t.post is None


def test_completed_defaults_to_now(clock: FakeClock) -> None:
    ctx = marked(clock, stream=False, marks={})
    assert compute_timings(ctx, now=ctx.received_at + 0.5).total == pytest.approx(0.5)


def test_fallback_uses_attempt_log_and_reports_failover(clock: FakeClock) -> None:
    ctx = marked(clock, stream=True, marks={"client_first_byte": 1.050, "completed": 2.000})
    start = ctx.received_at
    ctx.attempts = [
        AttemptRecord("anthropic/haiku", "anthropic", start + 0.010, 0.300, "fallback", "retryable", 529),
        AttemptRecord("openai/luna", "openai", start + 0.400, 1.590, "ok", ttft_s=0.600),
    ]
    t = compute_timings(ctx, now=999.0)
    assert t.pre_upstream == pytest.approx(0.010)
    assert t.failover == pytest.approx(0.390)
    assert t.upstream == pytest.approx(1.590)
    assert t.upstream_ttft == pytest.approx(0.600)
    assert t.ttft_added == pytest.approx(1.050 - 0.600 - 0.390)
    assert t.stream_tail == pytest.approx(0.010)
    assert t.overhead_phases()["failover"] == pytest.approx(0.390)


def test_cache_hit_and_block_have_no_upstream_phases(clock: FakeClock) -> None:
    hit = marked(
        clock, stream=True, marks={"upstream_start": 0.01, "client_first_byte": 0.004, "completed": 0.01}
    )
    hit.cache_status = "exact_hit"
    t = compute_timings(hit, now=0)
    assert t.ttft == pytest.approx(0.004)
    assert t.upstream is None
    assert t.overhead is None
    assert t.overhead_phases() == {}

    blocked = marked(clock, stream=False, marks={"completed": 0.003})
    assert compute_timings(blocked, now=0).overhead_phases() == {}


def test_inconsistent_marks_clamp_to_zero(clock: FakeClock) -> None:
    ctx = marked(clock, stream=False, marks={"upstream_start": 0.01, "upstream_end": 0.5, "completed": 0.4})
    t = compute_timings(ctx, now=0)
    assert t.post == 0.0
    assert t.overhead == 0.0


def test_final_attempt_prefers_last_ok() -> None:
    ok = AttemptRecord("a", "openai", 0, 1, "ok")
    failed = AttemptRecord("b", "openai", 1, 1, "fail")
    assert final_attempt([ok, failed]) is ok
    assert final_attempt([failed]) is failed
    assert final_attempt([]) is None
