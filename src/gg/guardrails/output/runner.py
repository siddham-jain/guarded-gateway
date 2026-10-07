"""non-stream output checks, and restore-only for cached responses"""

from dataclasses import dataclass
from typing import Any

from gg.core.context import RequestContext
from gg.core.guard_types import OutputVerdict, Verdict
from gg.core.schema import AssistantMessage, ChatResponse, Choice, ToolCall
from gg.guardrails.base import GuardContext, GuardFinding, Segment, SegmentKind
from gg.guardrails.engine import OUTPUT_DETECT, GuardChain, GuardrailEngine
from gg.guardrails.output.common import Finalizer, notable, report_restores, request_vault
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.redact import apply, marker, merge


def block_marker(policy: EffectivePolicy) -> dict[str, Any]:
    # stage, action and policy only: which guard fired never leaves the gateway
    return {"stage": "output", "action": "block", "policy": policy.ref.header()}


def _restore_message(msg: AssistantMessage, finalize: Finalizer) -> AssistantMessage:
    update: dict[str, Any] = {}
    if msg.content:
        update["content"] = finalize(msg.content, "content")
    if msg.refusal:
        update["refusal"] = finalize(msg.refusal, "refusal")
    if msg.tool_calls:
        update["tool_calls"] = tuple(
            c.model_copy(
                update={
                    "function": c.function.model_copy(
                        update={"arguments": finalize(c.function.arguments, "tool_args")}
                    )
                }
            )
            for c in msg.tool_calls
        )
    return msg.model_copy(update=update) if update else msg


def restore_response(response: ChatResponse, finalize: Finalizer) -> ChatResponse:
    choices = tuple(
        c.model_copy(update={"message": _restore_message(c.message, finalize)}) for c in response.choices
    )
    return response.model_copy(update={"choices": choices})


@dataclass(frozen=True, slots=True)
class OutputResult:
    response: ChatResponse
    # guarded content before restore (placeholder space), for post-hoc guards
    raw_text: str


class OutputGuardRunner:
    def __init__(self, engine: GuardrailEngine) -> None:
        self._engine = engine

    async def _check_text(
        self, chain: GuardChain, kind: SegmentKind, text: str, ctx: RequestContext, found: list[GuardFinding]
    ) -> str | None:
        """the text with redaction markers applied, or None when an enforce guard blocked it"""
        if not chain or not text:
            return text
        gctx = GuardContext(
            stage="output",
            request_id=ctx.request_id,
            segments=(Segment(index=0, role="assistant", kind=kind, msg=-1, text=text),),
            request=ctx.request,
            vault=request_vault(ctx),
            key=ctx.key,
        )
        decision = await self._engine.run(chain, gctx, tiers=OUTPUT_DETECT)
        found.extend(notable(decision.findings))
        if decision.blocked:
            return None
        return apply(text, merge(decision.redactions(enforced_only=True)), marker)

    async def _check_message(
        self, msg: AssistantMessage, policy: EffectivePolicy, ctx: RequestContext, found: list[GuardFinding]
    ) -> AssistantMessage | None:
        chain = policy.output_detectors(streaming=False)
        tool_chain = chain.select(lambda g: g.name in policy.doc.output.streaming.tool_args_guards)
        update: dict[str, Any] = {}
        parts: tuple[tuple[SegmentKind, str | None], ...] = (
            ("content", msg.content),
            ("refusal", msg.refusal),
        )
        for kind, text in parts:
            if text is None:
                continue
            checked = await self._check_text(chain, kind, text, ctx, found)
            if checked is None:
                return None
            update[kind] = checked
        calls: list[ToolCall] = []
        for call in msg.tool_calls or ():
            checked = await self._check_text(tool_chain, "tool_args", call.function.arguments, ctx, found)
            if checked is None:
                return None
            calls.append(
                call.model_copy(update={"function": call.function.model_copy(update={"arguments": checked})})
            )
        if calls:
            update["tool_calls"] = tuple(calls)
        return msg.model_copy(update=update)

    async def check(
        self, response: ChatResponse, ctx: RequestContext, policy: EffectivePolicy
    ) -> OutputResult:
        finalize = Finalizer(policy, ctx.request, request_vault(ctx))
        found: list[GuardFinding] = []
        choices: list[Choice] = []
        raw: list[str] = []
        blocked = False
        for choice in response.choices:
            checked = await self._check_message(choice.message, policy, ctx, found)
            if checked is not None:
                raw.append(checked.content or "")
            if checked is None:
                blocked = True
                message = AssistantMessage(content=None, refusal=policy.doc.response.messages.output_blocked)
                choices.append(Choice(index=choice.index, message=message, finish_reason="content_filter"))
            else:
                restored = _restore_message(checked, finalize)
                choices.append(choice.model_copy(update={"message": restored}))
        ctx.guard_findings.extend(found)
        report_restores(self._engine.metrics, request_vault(ctx))
        verdict = max((f.verdict for f in found), default=Verdict.ALLOW)
        ctx.output_verdict = OutputVerdict(verdict=verdict, findings=tuple(found))
        update: dict[str, Any] = {"choices": tuple(choices)}
        if blocked:
            update["gg_guardrail"] = block_marker(policy)
            ctx.response_headers["x-gg-guardrails"] = "blocked"
        return OutputResult(response.model_copy(update=update), "".join(raw))
