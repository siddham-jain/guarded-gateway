"""cross-user pii: cached replies hold placeholders, every hit is restored from its own request's vault"""

import pytest

from gg.cache.memory import InMemoryResponseCache
from gg.cache.stages import ExactCacheStage
from gg.core.cache_types import CacheState
from gg.core.clock import FakeClock, SystemClock
from gg.core.context import RequestContext
from gg.core.schema import ChatResponse
from gg.guardrails.output.stage import OutputGuardStage
from gg.guardrails.stages import InputGuardPreStage
from gg.guardrails.vault import GuardVault
from gg.pipeline.runner import Pipeline
from gg.pipeline.stage import PipelineResult
from tests.unit.cache.support import (
    RecordingHooks,
    Upstream,
    codec,
    ctx_for,
    drain,
    finish_request,
    response,
    stage_over,
)
from tests.unit.guardrails.support import engine, policy_set

USERS = [
    ("alice@corp.example", "bob@corp.example"),
    ("dana.smith@mail.example", "x.y@other.example"),
    ("Zoe+tag@sub.corp.example", "QA@corp.example"),
]


class EchoPlaceholders(Upstream):
    """the model only ever sees placeholders and echoes them back"""

    async def __call__(self, ctx: RequestContext) -> PipelineResult:
        text = ctx.request.last_user_text() or ""
        self.reply = response(f"Done: I emailed {text.split()[-1]} for you.")
        return await super().__call__(ctx)


def pipeline(stage: ExactCacheStage, upstream: Upstream) -> Pipeline:
    guards = engine()
    return Pipeline(
        [InputGuardPreStage(policy_set(), guards), OutputGuardStage(guards, clock=SystemClock()), stage],
        upstream,
        clock=SystemClock(),
    )


def ask(email: str, *, stream: bool = False) -> RequestContext:
    return ctx_for(messages=[{"role": "user", "content": f"please email {email}"}], stream=stream)


def content(result: PipelineResult) -> str:
    assert isinstance(result.response, ChatResponse)
    return result.response.choices[0].message.content or ""


@pytest.mark.parametrize(("email_a", "email_b"), USERS)
async def test_user_b_never_sees_user_a_values(email_a: str, email_b: str) -> None:
    backend = InMemoryResponseCache(codec(), FakeClock())
    hooks = RecordingHooks()
    upstream = EchoPlaceholders()
    run = pipeline(stage_over(backend, hooks, FakeClock()), upstream).run

    a = ask(email_a)
    first = await run(a)
    await finish_request(a)
    assert email_a in content(first)
    assert hooks.stores == [("exact", "stored", "ok")]

    b = ask(email_b)
    second = await run(b)
    await finish_request(b)
    assert b.cache_status == "exact_hit"
    assert upstream.calls == 1
    assert email_b in content(second)
    assert email_a not in content(second)

    stream_b = ask(email_b, stream=True)
    streamed = await run(stream_b)
    text = "".join(c.choices[0].delta.content or "" for c in await drain(streamed.stream) if c.choices)
    assert stream_b.cache_status == "exact_hit"
    assert email_b in text
    assert email_a not in text

    for raw in backend.raw_values():
        decoded = codec().decode(raw)
        stored = f"{decoded.content} {decoded.refusal}"
        assert email_a not in stored
        assert email_b not in stored
        assert "[EMAIL_1]" in stored


async def test_unresolvable_placeholder_is_a_miss() -> None:
    hooks = RecordingHooks()
    clock = FakeClock()
    backend = InMemoryResponseCache(codec(), clock)
    stage = stage_over(backend, hooks, clock)
    upstream = Upstream(response("Mail [EMAIL_1] today."))
    writer = ctx_for(messages=[{"role": "user", "content": "say hi"}])
    vault = GuardVault()
    vault.add("EMAIL", "w@corp.example")
    writer.vault = vault
    await stage(writer, upstream)
    await finish_request(writer)
    reader = ctx_for(messages=[{"role": "user", "content": "say hi"}])
    result = await stage(reader, upstream)
    assert result.source == "upstream"
    assert upstream.calls == 2
    assert hooks.lookups[-1] == ("exact", "miss")


async def test_vault_value_in_output_is_rejected() -> None:
    hooks = RecordingHooks()
    clock = FakeClock()
    backend = InMemoryResponseCache(codec(), clock)
    ctx = ctx_for(messages=[{"role": "user", "content": "say hi"}])
    vault = GuardVault()
    vault.add("EMAIL", "leak@corp.example")
    ctx.vault = vault
    # a mis-wired tap that hands the writer restored text
    await stage_over(backend, hooks, clock)(ctx, Upstream(response("Mail leak@corp.example today.")))
    await finish_request(ctx)
    assert hooks.stores == [("exact", "rejected", "vault_value_in_output")]
    assert backend.raw_values() == []


async def test_shadow_mode_pii_bypasses_and_writes_nothing() -> None:
    backend = InMemoryResponseCache(codec(), FakeClock())
    hooks = RecordingHooks()
    stage = stage_over(backend, hooks, FakeClock())
    ctx = ask("alice@corp.example")
    ctx.cache = CacheState(bypass_reason="unredacted_upstream", store=False)
    await stage(ctx, Upstream())
    await finish_request(ctx)
    assert ctx.response_headers["x-gg-cache"] == "BYPASS"
    assert hooks.bypasses == ["pii_unredacted"]
    assert backend.raw_values() == []
