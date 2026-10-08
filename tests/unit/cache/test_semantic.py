import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from gg.cache.config import CacheConfig
from gg.cache.embedders.hashing import HashingEmbedder
from gg.cache.setup import BuiltCache
from gg.core.clock import FakeClock
from gg.core.context import RequestContext
from gg.core.errors import GGError
from gg.pipeline.probes import ConcurrentProbesStage, ProbeOutcome, Reject
from gg.pipeline.runner import Pipeline
from gg.pipeline.stage import PipelineResult
from tests.unit.cache.support import RecordingHooks, Upstream, built, ctx_for, finish_request, response

SEMANTIC_KEY = {"cache": {"semantic": True}}
ANCHOR = "What is the capital of France?"


class CountingEmbedder(HashingEmbedder):
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        super().__init__(64)
        self.calls = 0
        self.fail = fail
        self.delay = delay

    async def embed(self, texts: Sequence[str], /) -> list[list[float]]:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("onnx exploded")
        return await super().embed(texts)


class BlockingGuard:
    name = "guard_t2"
    precedence = 10

    async def __call__(self, ctx: RequestContext, /) -> ProbeOutcome:
        await asyncio.sleep(0.01)
        return Reject(GGError("blocked by tier 2"))


def setup(
    embedder: CountingEmbedder | None = None, hooks: RecordingHooks | None = None
) -> tuple[BuiltCache, CountingEmbedder]:
    embedder = embedder or CountingEmbedder()
    cache = built(embedder=embedder, hooks=hooks, clock=FakeClock())
    return cache, embedder


def pipeline(cache: BuiltCache, upstream: Upstream, *extra_probes: Any) -> Pipeline:
    assert cache.semantic_probe is not None
    probes = ConcurrentProbesStage([cache.semantic_probe, *extra_probes], FakeClock())
    return Pipeline([cache.exact_stage, probes], upstream, clock=FakeClock())


def ask(text: str, **request: Any) -> RequestContext:
    return ctx_for(key=SEMANTIC_KEY, messages=[{"role": "user", "content": text}], **request)


async def go(p: Pipeline, ctx: RequestContext) -> PipelineResult:
    result = await p.run(ctx)
    await finish_request(ctx)
    return result


async def test_paraphrase_is_a_semantic_hit() -> None:
    hooks = RecordingHooks()
    cache, _ = setup(hooks=hooks)
    upstream = Upstream(response("Paris."))
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR))
    assert ("semantic", "stored", "ok") in hooks.stores
    ctx = ask("What is the capital city of France?")
    result = await go(p, ctx)
    assert upstream.calls == 1
    assert result.source == "semantic_cache"
    assert ctx.cache_status == "semantic_hit"
    assert ctx.response_headers["x-gg-cache"] == "SEMANTIC_HIT"
    assert ctx.response_headers["x-gg-cache-distance"] == "0.0564"
    assert ctx.cache is not None
    assert ctx.cache.semantic_distance == pytest.approx(0.0564, abs=1e-4)
    assert result.response is not None
    assert result.response.choices[0].message.content == "Paris."
    assert ("semantic", "hit") in hooks.lookups
    assert hooks.distances[-1][0] == "hit"


async def test_semantic_hit_streams() -> None:
    cache, _ = setup()
    upstream = Upstream(response("Paris."))
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR))
    ctx = ask("What is the capital city of France?", stream=True)
    result = await p.run(ctx)
    assert result.stream is not None
    text = "".join([c.choices[0].delta.content or "" async for c in result.stream if c.choices])
    assert text == "Paris."


async def test_far_question_is_a_miss_with_observed_distance() -> None:
    hooks = RecordingHooks()
    cache, _ = setup(hooks=hooks)
    upstream = Upstream()
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR))
    await go(p, ask("What is the capital of Germany?"))
    assert upstream.calls == 2
    assert hooks.distances[-1][0] == "miss"
    assert hooks.distances[-1][1] > 0.08


async def test_number_change_never_matches() -> None:
    cache, _ = setup()
    upstream = Upstream()
    p = pipeline(cache, upstream)
    await go(p, ask("What is 17 times 23?"))
    ctx = ask("What is 17 times 24?")
    await go(p, ctx)
    assert upstream.calls == 2
    assert ctx.cache_status == "miss"


async def test_request_threshold_only_tightens() -> None:
    cache, _ = setup()
    upstream = Upstream()
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR))
    strict = ask("What is the capital city of France?", gg={"cache_threshold": 0.01})
    await go(p, strict)
    assert strict.cache_status == "miss"
    loose = ask("Which city is the capital of France?", gg={"cache_threshold": 0.9})
    await go(p, loose)
    assert loose.cache_status == "miss"
    assert upstream.calls == 3


async def test_tier_two_block_beats_a_semantic_hit() -> None:
    cache, _ = setup()
    upstream = Upstream()
    await go(pipeline(cache, upstream), ask(ANCHOR))
    guarded = pipeline(cache, upstream, BlockingGuard())
    with pytest.raises(GGError):
        await guarded.run(ask("What is the capital city of France?"))
    assert upstream.calls == 1


async def test_ineligible_requests_skip_the_embedder() -> None:
    cache, embedder = setup()
    upstream = Upstream()
    p = pipeline(cache, upstream)
    await go(p, ctx_for(messages=[{"role": "user", "content": ANCHOR}]))
    await go(p, ask(ANCHOR, gg={"semantic_cache": False}))
    await go(p, ask("hi"))
    assert embedder.calls == 0


async def test_writer_reuses_the_probe_vector() -> None:
    cache, embedder = setup()
    await go(pipeline(cache, Upstream()), ask(ANCHOR))
    assert embedder.calls == 1


@pytest.mark.parametrize(("failure", "result"), [({"fail": True}, "error"), ({"delay": 1.0}, "timeout")])
async def test_embedder_trouble_is_a_miss(failure: dict[str, Any], result: str) -> None:
    hooks = RecordingHooks()
    cache, _ = setup(CountingEmbedder(**failure), hooks=hooks)
    upstream = Upstream()
    ctx = ask(ANCHOR)
    assert (await pipeline(cache, upstream).run(ctx)).source == "upstream"
    assert ("semantic", result) in hooks.lookups
    assert ctx.cache_status == "miss"


async def test_expired_value_behind_a_vector_is_a_miss() -> None:
    clock = FakeClock()
    embedder = CountingEmbedder()
    cache = built(embedder=embedder, clock=clock)
    upstream = Upstream()
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR, gg={"cache_ttl_s": 60}))
    clock.advance(61)
    ctx = ask("What is the capital city of France?")
    await go(p, ctx)
    assert ctx.cache_status == "miss"
    assert upstream.calls == 2


async def test_semantic_scope_isolation() -> None:
    cache, _ = setup()
    upstream = Upstream()
    p = pipeline(cache, upstream)
    owner = ctx_for(key={"id": "key-a", **SEMANTIC_KEY}, messages=[{"role": "user", "content": ANCHOR}])
    await go(p, owner)
    other = ctx_for(
        key={"id": "key-b", **SEMANTIC_KEY},
        messages=[{"role": "user", "content": "What is the capital city of France?"}],
    )
    await go(p, other)
    assert other.cache_status == "miss"
    assert upstream.calls == 2


class Verifier:
    def __init__(self, answer: bool | Exception) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def same_answer(self, cached_prompt: str, new_prompt: str, /) -> bool:
        self.asked.append((cached_prompt, new_prompt))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


async def paraphrase_with(verifier: Verifier) -> tuple[PipelineResult, Upstream, RecordingHooks]:
    hooks = RecordingHooks()
    cache = built(embedder=CountingEmbedder(), hooks=hooks, clock=FakeClock(), verifier=verifier)
    upstream = Upstream(response("Paris."))
    p = pipeline(cache, upstream)
    await go(p, ask(ANCHOR))
    return await go(p, ask("What is the capital city of France?")), upstream, hooks


async def test_a_verified_match_is_served() -> None:
    verifier = Verifier(True)
    result, upstream, _ = await paraphrase_with(verifier)
    assert result.source == "semantic_cache"
    assert upstream.calls == 1
    assert verifier.asked == [(ANCHOR, "What is the capital city of France?")]


@pytest.mark.parametrize("answer", [False, RuntimeError("jev is down")])
async def test_a_match_the_verifier_rejects_or_cannot_judge_goes_upstream(answer: bool | Exception) -> None:
    result, upstream, hooks = await paraphrase_with(Verifier(answer))
    assert result.source == "upstream"
    assert upstream.calls == 2
    assert ("semantic", "miss") in hooks.lookups
    assert hooks.distances[-1][0] == "miss"


async def test_the_verifier_is_not_asked_outside_the_threshold() -> None:
    verifier = Verifier(True)
    cache = built(embedder=CountingEmbedder(), clock=FakeClock(), verifier=verifier)
    p = pipeline(cache, Upstream())
    await go(p, ask(ANCHOR))
    await go(p, ask("How do I undo the last git commit?"))
    assert verifier.asked == []


def test_a_configured_verifier_that_is_missing_disables_semantic_lookups() -> None:
    semantic = {
        "embedder": {"provider": "hashing", "name": "hashing", "dim": 64},
        "verifier": {"type": "jev"},
    }
    cfg = CacheConfig.model_validate({"semantic": semantic})
    assert built(cfg, embedder=CountingEmbedder()).semantic_probe is None
    assert built(cfg, embedder=CountingEmbedder(), verifier=Verifier(True)).semantic_probe is not None
