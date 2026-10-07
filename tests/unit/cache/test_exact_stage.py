import asyncio
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from gg.cache.base import CachedResponse
from gg.core.aio import TaskSupervisor
from gg.core.clock import FakeClock
from gg.core.guard_types import OutputVerdict, Verdict
from gg.core.schema import FunctionCall, ToolCall, to_wire
from gg.core.usage import UsageRecord
from gg.guardrails.base import POLICY_REF, PolicyRef
from gg.pipeline.stage import PipelineResult
from tests.unit.cache.support import (
    BrokenBackend,
    RecordingHooks,
    Upstream,
    built,
    config,
    ctx_for,
    drain,
    response,
    run,
    stage_over,
    text_of,
)

PROMPT = [{"role": "user", "content": "What is the capital of France?"}]


async def test_second_identical_request_is_a_hit_without_upstream() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    upstream = Upstream()
    first, second = ctx_for(messages=PROMPT), ctx_for(messages=PROMPT)
    await run(cache, first, upstream)
    assert first.cache_status == "miss"
    assert first.response_headers["x-gg-cache"] == "MISS"
    result = await run(cache, second, upstream)
    assert upstream.calls == 1
    assert result.source == "exact_cache"
    assert second.cache_status == "exact_hit"
    assert text_of(result) == "Paris."
    headers = second.response_headers
    assert headers["x-gg-cache"] == "HIT"
    assert headers["x-gg-cost-usd"] == "0"
    assert headers["x-gg-provider"] == "mock"
    assert headers["x-gg-model"] == "mock/echo"
    assert headers["x-gg-cache-age"] == "0"
    assert result.response is not None
    assert result.response.id == "chatcmpl-test"
    assert result.response.usage is not None
    assert result.response.usage.completion_tokens == 7
    assert ("exact", "stored", "ok") in hooks.stores
    assert hooks.lookups == [("exact", "miss"), ("exact", "hit")]


async def test_savings_are_booked_through_the_pricer() -> None:
    hooks = RecordingHooks()
    priced: list[UsageRecord] = []

    def pricer(usage: UsageRecord) -> float:
        priced.append(usage)
        return 0.0125

    cache = built(hooks=hooks, pricer=pricer)
    await run(cache, ctx_for(messages=PROMPT), Upstream())
    hit = ctx_for(messages=PROMPT)
    await run(cache, hit, Upstream())
    assert hooks.saved == [("exact", 0.0125)]
    assert ("exact", "input", 12) in hooks.tokens
    assert hit.response_headers["x-gg-cost-saved-usd"] == "0.012500"
    assert priced[-1].deployment_id == "mock/echo"


async def test_falls_back_to_stored_cost_when_no_longer_priced() -> None:
    hooks = RecordingHooks()
    prices = iter([0.5, None])
    cache = built(hooks=hooks, pricer=lambda usage: next(prices))
    await run(cache, ctx_for(messages=PROMPT), Upstream())
    await run(cache, ctx_for(messages=PROMPT), Upstream())
    assert hooks.saved == [("exact", 0.5)]


async def test_stream_hit_replays_valid_openai_chunks_with_usage_iff_requested() -> None:
    cache = built()
    upstream = Upstream(parts=("Par", "is."))
    await run(cache, ctx_for(messages=PROMPT, stream=True), upstream)
    for wants_usage in (True, False):
        ctx = ctx_for(messages=PROMPT, stream=True, stream_options={"include_usage": wants_usage})
        result = await cache.exact_stage(ctx, upstream)
        assert result.source == "exact_cache"
        chunks = await drain(result.stream)
        parsed = [ChatCompletionChunk.model_validate(to_wire(c)) for c in chunks]
        text = "".join(c.choices[0].delta.content or "" for c in parsed if c.choices)
        assert text == "Paris."
        assert parsed[0].choices[0].delta.role == "assistant"
        finishes = [c.choices[0].finish_reason for c in parsed if c.choices and c.choices[0].finish_reason]
        assert finishes == ["stop"]
        usage = [c.usage for c in parsed if c.usage is not None]
        assert len(usage) == (1 if wants_usage else 0)
        assert all(c.id == parsed[0].id for c in parsed)
    assert upstream.calls == 1


async def test_stream_entry_serves_json_and_json_entry_serves_stream() -> None:
    cache = built()
    upstream = Upstream()
    await run(cache, ctx_for(messages=PROMPT, stream=True), upstream)
    result = await run(cache, ctx_for(messages=PROMPT), upstream)
    assert text_of(result) == "Paris."
    other = [{"role": "user", "content": "And of Spain?"}]
    await run(cache, ctx_for(messages=other), Upstream(response("Madrid.")))
    replay = await cache.exact_stage(ctx_for(messages=other, stream=True), upstream)
    chunks = await drain(replay.stream)
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "Madrid."
    assert upstream.calls == 1


async def test_disconnected_stream_is_not_stored() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    upstream = Upstream(parts=("Par", "is."))
    ctx = ctx_for(messages=PROMPT, stream=True)
    result = await cache.exact_stage(ctx, upstream)
    assert result.stream is not None
    await anext(result.stream)
    await result.stream.aclose()
    ctx.outcome = "client_disconnected"
    await ctx.finalizers.run(timeout_s=5)
    assert ("exact", "skipped", "disconnect") in hooks.stores
    await run(cache, ctx_for(messages=PROMPT), upstream)
    assert upstream.calls == 2


async def test_unfinished_stream_is_not_stored_even_if_marked_completed() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    ctx = ctx_for(messages=PROMPT, stream=True)
    result = await cache.exact_stage(ctx, Upstream())
    assert result.stream is not None
    await anext(result.stream)
    await result.stream.aclose()
    ctx.outcome = "completed"
    await ctx.finalizers.run(timeout_s=5)
    assert hooks.stores == [("exact", "skipped", "error")]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (response(finish="content_filter"), "finish_reason"),
        (
            response(
                None,
                finish="tool_calls",
                tool_calls=(ToolCall(id="t", function=FunctionCall(name="f", arguments="{}")),),
            ),
            "tool_calls",
        ),
        (response(""), "empty"),
    ],
)
async def test_unstorable_replies(reply: Any, reason: str) -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    await run(cache, ctx_for(messages=PROMPT), Upstream(reply))
    assert hooks.stores == [("exact", "skipped", reason)]


async def test_length_finish_is_stored() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    await run(cache, ctx_for(messages=PROMPT), Upstream(response("Par", finish="length")))
    assert hooks.stores == [("exact", "stored", "ok")]


@pytest.mark.parametrize("verdict", [Verdict.FLAG, Verdict.REDACT, Verdict.BLOCK])
async def test_non_allow_output_verdict_is_not_stored(verdict: Verdict) -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    ctx = ctx_for(messages=PROMPT)
    ctx.set(POLICY_REF, PolicyRef("default", "1", "abc"))
    result = await cache.exact_stage(ctx, Upstream())
    ctx.output_verdict = OutputVerdict(verdict=verdict)
    ctx.outcome = "completed"
    await ctx.finalizers.run(timeout_s=5)
    assert result.source == "upstream"
    assert hooks.stores == [("exact", "skipped", "guard_blocked")]


async def test_guarded_request_without_a_settled_verdict_is_not_stored() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    ctx = ctx_for(messages=PROMPT)
    ctx.set(POLICY_REF, PolicyRef("default", "1", "abc"))
    await run(cache, ctx, Upstream())
    assert hooks.stores == [("exact", "skipped", "guard_pending")]


async def test_write_waits_for_post_hoc_guards() -> None:
    hooks = RecordingHooks()
    supervisor = TaskSupervisor()
    cache = built(config(post_hoc_wait_s=1.0), hooks=hooks, supervisor=supervisor)
    for final in (Verdict.ALLOW, Verdict.BLOCK):
        ctx = ctx_for(messages=[{"role": "user", "content": f"question {final.name}"}])
        ctx.set(POLICY_REF, PolicyRef("default", "1", "abc"))
        await cache.exact_stage(ctx, Upstream())
        ctx.output_verdict = OutputVerdict(verdict=Verdict.ALLOW, post_hoc_pending=True)
        ctx.outcome = "completed"
        await ctx.finalizers.run(timeout_s=5)
        assert hooks.stores == []
        ctx.output_verdict = OutputVerdict(verdict=final)
        await supervisor.drain(timeout_s=5)
        expected = (
            ("exact", "stored", "ok") if final is Verdict.ALLOW else ("exact", "skipped", "guard_blocked")
        )
        assert hooks.stores == [expected]
        hooks.stores.clear()


async def test_bypass_header_and_reason() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    upstream = Upstream()
    ctx = ctx_for(messages=PROMPT, temperature=0.9)
    await run(cache, ctx, upstream)
    await run(cache, ctx_for(messages=PROMPT, temperature=0.9), upstream)
    assert upstream.calls == 2
    assert ctx.cache_status == "bypass"
    assert ctx.response_headers["x-gg-cache"] == "BYPASS"
    assert ctx.cache is not None
    assert ctx.cache.bypass_reason == "sampled"
    assert hooks.bypasses == ["sampled", "sampled"]
    assert hooks.stores == []


async def test_refresh_skips_lookup_but_overwrites_and_no_store_reads_only() -> None:
    cache = built()
    await run(cache, ctx_for(messages=PROMPT), Upstream(response("old")))
    refreshed = ctx_for(messages=PROMPT, gg={"cache": "refresh"})
    assert text_of(await run(cache, refreshed, Upstream(response("new")))) == "new"
    assert refreshed.cache_status == "miss"
    assert text_of(await run(cache, ctx_for(messages=PROMPT), Upstream())) == "new"
    no_store = ctx_for(messages=[{"role": "user", "content": "uncached question"}], gg={"cache": "no_store"})
    upstream = Upstream()
    await run(cache, no_store, upstream)
    await run(cache, ctx_for(messages=[{"role": "user", "content": "uncached question"}]), upstream)
    assert upstream.calls == 2
    hit = ctx_for(messages=PROMPT, gg={"cache": "no_store"})
    assert (await run(cache, hit, Upstream())).source == "exact_cache"


async def test_scope_isolation_between_keys_and_global() -> None:
    cache = built()
    upstream = Upstream()
    await run(cache, ctx_for(messages=PROMPT, key={"id": "key-a"}), upstream)
    await run(cache, ctx_for(messages=PROMPT, key={"id": "key-b"}), upstream)
    assert upstream.calls == 2
    await run(cache, ctx_for(messages=PROMPT, key={"id": "key-g1", "cache": {"scope": "global"}}), upstream)
    hit = ctx_for(messages=PROMPT, key={"id": "key-g2", "cache": {"scope": "global"}})
    assert (await run(cache, hit, upstream)).source == "exact_cache"
    assert upstream.calls == 3


async def test_policy_change_invalidates() -> None:
    cache = built()
    upstream = Upstream()
    for policy_hash in ("v1", "v1", "v2"):
        ctx = ctx_for(messages=PROMPT)
        ctx.set(POLICY_REF, PolicyRef("default", "1", policy_hash))
        result = await cache.exact_stage(ctx, upstream)
        ctx.output_verdict = OutputVerdict(verdict=Verdict.ALLOW) if result.source == "upstream" else None
        ctx.outcome = "completed"
        await ctx.finalizers.run(timeout_s=5)
    assert upstream.calls == 2


async def test_alias_revision_change_invalidates() -> None:
    revision = {"value": "r1"}
    cache = built(alias_revision=lambda ctx: revision["value"])
    upstream = Upstream()
    await run(cache, ctx_for(messages=PROMPT), upstream)
    await run(cache, ctx_for(messages=PROMPT), upstream)
    revision["value"] = "r2"
    await run(cache, ctx_for(messages=PROMPT), upstream)
    assert upstream.calls == 2


async def test_single_flight_makes_one_upstream_call() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    upstream = Upstream(delay=0.05)
    results = await asyncio.gather(*(run(cache, ctx_for(messages=PROMPT), upstream) for _ in range(5)))
    assert upstream.calls == 1
    assert sorted(r.source for r in results) == ["exact_cache"] * 4 + ["upstream"]
    assert hooks.flights.count("leader") == 1
    assert hooks.flights.count("waited_hit") == 4


async def test_single_flight_followers_proceed_when_the_leader_fails() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    failing = Upstream(delay=0.03, fail=RuntimeError("upstream 500"))
    working = Upstream()
    leader_ctx = ctx_for(messages=PROMPT)

    async def leader() -> None:
        with pytest.raises(RuntimeError):
            await cache.exact_stage(leader_ctx, failing)
        leader_ctx.outcome = "upstream_error"
        await leader_ctx.finalizers.run(timeout_s=5)

    async def follower() -> PipelineResult:
        await asyncio.sleep(0.005)
        return await run(cache, ctx_for(messages=PROMPT), working)

    _, result = await asyncio.gather(leader(), follower())
    assert result.source == "upstream"
    assert working.calls == 1
    assert hooks.flights == ["leader", "waited_miss"]


async def test_single_flight_follower_gives_up_after_the_wait() -> None:
    hooks = RecordingHooks()
    cache = built(config(singleflight_wait_s=0.05, singleflight_poll_s=0.01), hooks=hooks)
    slow = Upstream(delay=0.3)
    fast = Upstream()

    async def follower() -> PipelineResult:
        await asyncio.sleep(0.005)
        return await run(cache, ctx_for(messages=PROMPT), fast)

    _, result = await asyncio.gather(run(cache, ctx_for(messages=PROMPT), slow), follower())
    assert result.source == "upstream"
    assert "waited_timeout" in hooks.flights


async def test_streams_do_not_wait_on_single_flight() -> None:
    hooks = RecordingHooks()
    cache = built(hooks=hooks)
    await asyncio.gather(
        *(run(cache, ctx_for(messages=PROMPT, stream=True), Upstream(delay=0.02)) for _ in range(2))
    )
    assert hooks.flights == []


async def test_backend_errors_fail_open_then_the_breaker_bypasses() -> None:
    hooks = RecordingHooks()
    clock = FakeClock()
    backend = BrokenBackend()
    stage = stage_over(backend, hooks, clock)
    upstream = Upstream()
    for _ in range(3):
        ctx = ctx_for(messages=PROMPT)
        result = await stage(ctx, upstream)
        ctx.outcome = "completed"
        await ctx.finalizers.run(timeout_s=5)
        assert result.source == "upstream"
    assert upstream.calls == 3
    assert ("exact", "error") in hooks.lookups
    assert ("exact", "error", "backend") in hooks.stores
    calls = backend.calls
    ctx = ctx_for(messages=PROMPT)
    await stage(ctx, upstream)
    assert ctx.response_headers["x-gg-cache"] == "BYPASS"
    assert ctx.cache is not None
    assert ctx.cache.bypass_reason == "backend_down"
    assert backend.calls == calls
    clock.advance(11)
    await stage(ctx_for(messages=PROMPT), upstream)
    assert backend.calls > calls


async def test_slow_backend_counts_as_timeout() -> None:
    class SlowBackend(BrokenBackend):
        async def get(self, key: str, /) -> CachedResponse | None:
            await asyncio.sleep(1)
            return None

    hooks = RecordingHooks()
    stage = stage_over(SlowBackend(), hooks, FakeClock())
    result = await stage(ctx_for(messages=PROMPT), Upstream())
    assert result.source == "upstream"
    assert hooks.lookups == [("exact", "timeout")]
