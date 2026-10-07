"""windowed output checking over a chunk stream.

text is released only after the windowed guards vetted it. each window is checked together with an
overlap tail of already released text, and a trailing suffix that may be an unfinished secret, pii value
or placeholder is held back until later text resolves it.
"""

import re
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field

from gg.core.clock import Clock
from gg.core.context import RequestContext
from gg.core.errors import GuardrailBlockedError
from gg.core.guard_types import OutputVerdict, Verdict
from gg.core.schema import ChatChunk, ChunkChoice, Delta, FinishReason, FunctionCallDelta, ToolCallDelta
from gg.guardrails.base import GuardContext, GuardFinding, Holdback, Redaction, Segment, SegmentKind
from gg.guardrails.engine import OUTPUT_DETECT, GuardChain, GuardrailEngine
from gg.guardrails.output.common import Finalizer, notable, report_restores, request_vault
from gg.guardrails.output.runner import block_marker
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.redact import apply, marker, merge
from gg.guardrails.rules import RUN_CONTINUATION


@dataclass(slots=True)
class _Channel:
    choice: int
    kind: SegmentKind
    tool_index: int = -1
    pending: str = ""
    tail: str = ""
    windows: int = 0
    continuation: re.Pattern[str] | None = None
    tool_id: str | None = None
    tool_name: str | None = None
    header_sent: bool = False


@dataclass(slots=True)
class _Choice:
    role_pending: bool = False
    finished: bool = False
    channels: dict[tuple[SegmentKind, int], _Channel] = field(default_factory=lambda: {})


async def _aclose(stream: AsyncIterator[ChatChunk]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        await aclose()


class StreamGuard:
    def __init__(
        self,
        engine: GuardrailEngine,
        policy: EffectivePolicy,
        ctx: RequestContext,
        *,
        clock: Clock,
        detect: bool = True,
    ) -> None:
        self._engine = engine
        self._policy = policy
        self._ctx = ctx
        self._clock = clock
        self._cfg = policy.doc.output.streaming
        self._chain = policy.output_detectors(streaming=True) if detect else GuardChain()
        self._tool_chain = self._chain.select(lambda g: g.name in self._cfg.tool_args_guards)
        self._vault = request_vault(ctx)
        self._finalize = Finalizer(policy, ctx.request, self._vault)
        self._choices: dict[int, _Choice] = {}
        self._template: ChatChunk | None = None
        self._findings: list[GuardFinding] = []
        self._spent = 0.0
        self._error: GuardrailBlockedError | None = None
        self._emitted = False
        self._reply: list[str] = []
        self.blocked = False

    async def guard(self, upstream: AsyncIterator[ChatChunk]) -> AsyncGenerator[ChatChunk]:
        usage: list[ChatChunk] = []
        try:
            while not self.blocked:
                try:
                    chunk = await anext(upstream)
                except StopAsyncIteration:
                    break
                except Exception:
                    # a committed upstream failure: release what is already vetted, then let the error through
                    out = await self._flush()
                    for c in out:
                        yield c
                    if self.blocked:
                        break
                    raise
                if not chunk.choices:
                    usage.append(chunk)
                    continue
                out = await self._on_chunk(chunk)
                if self.blocked:
                    # cancel upstream generation before the client sees the end of the stream
                    await _aclose(upstream)
                for c in out:
                    yield c
            if not self.blocked:
                for c in await self._flush():
                    yield c
                for c in usage:
                    yield c
            if self._error is not None:
                raise self._error
        finally:
            await _aclose(upstream)
            self._settle()

    def _choice(self, index: int) -> _Choice:
        return self._choices.setdefault(index, _Choice())

    def _channel(self, index: int, kind: SegmentKind, tool_index: int = -1) -> _Channel:
        channels = self._choice(index).channels
        key = (kind, tool_index)
        if key not in channels:
            channels[key] = _Channel(choice=index, kind=kind, tool_index=tool_index)
        return channels[key]

    async def _on_chunk(self, chunk: ChatChunk) -> list[ChatChunk]:
        self._template = chunk
        out: list[ChatChunk] = []
        for choice in chunk.choices:
            state = self._choice(choice.index)
            delta = choice.delta
            if delta.role:
                state.role_pending = True
            if delta.content:
                out += await self._feed(self._channel(choice.index, "content"), delta.content)
            if delta.refusal:
                out += await self._feed(self._channel(choice.index, "refusal"), delta.refusal)
            for tc in delta.tool_calls or ():
                ch = self._channel(choice.index, "tool_args", tc.index)
                ch.tool_id = tc.id or ch.tool_id
                if tc.function is not None:
                    ch.tool_name = tc.function.name or ch.tool_name
                    if tc.function.arguments:
                        out += await self._feed(ch, tc.function.arguments)
            if self.blocked:
                return out + self._abort()
            if choice.finish_reason is not None:
                out += await self._finish(choice.index)
                if self.blocked:
                    return out + self._abort()
                out.append(self._finish_chunk(choice.index, choice.finish_reason))
        return out

    async def _feed(self, ch: _Channel, text: str) -> list[ChatChunk]:
        if self.blocked:
            return []
        ch.pending += text
        threshold = self._cfg.first_window_chars if ch.windows == 0 else self._cfg.window_chars
        return await self._check(ch, final=False) if len(ch.pending) >= threshold else []

    async def _finish(self, index: int) -> list[ChatChunk]:
        state = self._choice(index)
        out: list[ChatChunk] = []
        for ch in state.channels.values():
            out += await self._check(ch, final=True)
            if self.blocked:
                return out
            if ch.kind == "tool_args" and not ch.header_sent:
                out.append(self._release(ch, ""))
        state.finished = True
        return out

    async def _flush(self) -> list[ChatChunk]:
        out: list[ChatChunk] = []
        for index, state in self._choices.items():
            if not state.finished:
                out += await self._finish(index)
                if self.blocked:
                    return out + self._abort()
        return out

    async def _detect(self, ch: _Channel, text: str, final: bool) -> list[Redaction] | None:
        chain = self._tool_chain if ch.kind == "tool_args" else self._chain
        if not chain:
            return []
        gctx = GuardContext(
            stage="output",
            request_id=self._ctx.request_id,
            segments=(Segment(index=0, role="assistant", kind=ch.kind, msg=-1, text=text),),
            request=self._ctx.request,
            vault=self._vault,
            key=self._ctx.key,
            is_final=final,
        )
        start = self._clock.monotonic()
        decision = await self._engine.run(chain, gctx, tiers=OUTPUT_DETECT)
        self._spent += self._clock.monotonic() - start
        self._findings += notable(decision.findings)
        if decision.blocked:
            return None
        return merge(decision.redactions(enforced_only=True))

    def _hold(self, ch: _Channel, text: str) -> int:
        chain = self._tool_chain if ch.kind == "tool_args" else self._chain
        holds = [self._vault.partial_suffix_len(text)]
        holds += [g.guard.holdback(text) for g in chain.guards if isinstance(g.guard, Holdback)]
        return min(max(holds), self._cfg.holdback_max_chars, len(ch.pending))

    async def _check(self, ch: _Channel, *, final: bool) -> list[ChatChunk]:
        if ch.continuation is not None:
            # the rest of a secret that was already replaced by a marker in an earlier window
            m = ch.continuation.match(ch.pending)
            consumed = m.end() if m else 0
            if consumed == len(ch.pending) and not final:
                ch.pending = ""
                return []
            ch.pending = ch.pending[consumed:]
            ch.continuation = None
        if not ch.pending:
            return []
        text = ch.tail + ch.pending
        base = len(ch.tail)
        found = await self._detect(ch, text, final)
        if found is None:
            self.blocked = True
            return []
        spans = [s for s in found if s.end > base]
        cut = len(text) if final else len(text) - self._hold(ch, text)
        continuation: str | None = None
        cap = self._cfg.holdback_max_chars
        for span in spans:
            if final or span.start >= cut:
                continue
            if span.end == len(text):
                # the match may still grow: hold it, or if too long to hold, redact now and swallow the rest
                if len(text) - span.start <= cap and span.start >= base:
                    cut = span.start
                else:
                    cut = len(text)
                    continuation = span.continuation or RUN_CONTINUATION
            elif span.start < cut < span.end:
                cut = span.end
        cut = max(cut, base)
        released = [
            Redaction(0, max(s.start, base) - base, min(s.end, cut) - base, s.label)
            for s in spans
            if s.start < cut
        ]
        raw = apply(text[base:cut], released, marker)
        ch.pending = text[cut:]
        overlap = self._cfg.overlap_chars
        ch.tail = (ch.tail + raw)[-overlap:] if overlap else ""
        ch.windows += 1
        if continuation is not None:
            ch.continuation = re.compile(continuation)
        if not raw:
            return []
        if ch.kind == "content":
            self._reply.append(raw)
        return [self._release(ch, self._finalize(raw, ch.kind))]

    def _role(self, index: int) -> Delta:
        state = self._choice(index)
        if state.role_pending:
            state.role_pending = False
            return Delta(role="assistant")
        return Delta()

    def _emit(self, index: int, delta: Delta, finish: FinishReason | None = None) -> ChatChunk:
        t = self._template
        self._emitted = True
        return ChatChunk(
            id=t.id if t else "",
            created=t.created if t else 0,
            model=t.model if t else "",
            system_fingerprint=t.system_fingerprint if t else None,
            choices=(ChunkChoice(index=index, delta=delta, finish_reason=finish),),
        )

    def _release(self, ch: _Channel, text: str) -> ChatChunk:
        role = self._role(ch.choice).role
        if ch.kind == "content":
            return self._emit(ch.choice, Delta(role=role, content=text))
        if ch.kind == "refusal":
            return self._emit(ch.choice, Delta(role=role, refusal=text))
        first = not ch.header_sent
        ch.header_sent = True
        call = ToolCallDelta(
            index=ch.tool_index,
            id=ch.tool_id if first else None,
            type="function" if first else None,
            function=FunctionCallDelta(name=ch.tool_name if first else None, arguments=text),
        )
        return self._emit(ch.choice, Delta(role=role, tool_calls=(call,)))

    def _finish_chunk(self, index: int, reason: FinishReason) -> ChatChunk:
        return self._emit(index, self._role(index), reason)

    def _abort(self) -> list[ChatChunk]:
        self._ctx.outcome = "guard_aborted"
        self._ctx.response_headers["x-gg-guardrails"] = "blocked"
        open_choices = sorted(i for i, s in self._choices.items() if not s.finished) or [0]
        if self._cfg.abort == "error_frame":
            message = self._policy.doc.response.messages.output_blocked
            self._error = GuardrailBlockedError(message, headers={"x-gg-guardrail-stage": "output"})
            # commit the 200 first so the block arrives as an in-stream error event, never as a 4xx
            if self._emitted:
                return []
            return [self._emit(i, self._role(i)) for i in open_choices]
        t = self._template
        choices = tuple(
            ChunkChoice(index=i, delta=self._role(i), finish_reason="content_filter") for i in open_choices
        )
        for i in open_choices:
            self._choice(i).finished = True
        chunk = ChatChunk(
            id=t.id if t else "", created=t.created if t else 0, model=t.model if t else "", choices=choices
        )
        return [chunk.model_copy(update={"gg_guardrail": block_marker(self._policy)})]

    def _settle(self) -> None:
        self._ctx.guard_findings.extend(self._findings)
        verdict = (
            Verdict.BLOCK if self.blocked else max((f.verdict for f in self._findings), default=Verdict.ALLOW)
        )
        self._ctx.output_verdict = OutputVerdict(verdict=verdict, findings=tuple(self._findings))
        self._ctx.reply_text = "".join(self._reply)
        self._ctx.timings.record("guard_out", self._spent)
        report_restores(self._engine.metrics, self._vault)
