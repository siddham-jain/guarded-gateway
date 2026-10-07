from collections.abc import Iterable

from gg.core.context import RequestContext
from gg.core.guard_types import Verdict
from gg.core.schema import ChatRequest
from gg.guardrails.base import GuardFinding, GuardMetrics, Restorer, SegmentKind
from gg.guardrails.policy.effective import EffectivePolicy
from gg.guardrails.vault import GuardVault


def request_vault(ctx: RequestContext) -> GuardVault:
    if not isinstance(ctx.vault, GuardVault):
        ctx.vault = GuardVault.adopt(ctx.vault)
    return ctx.vault


def json_mode(request: ChatRequest) -> bool:
    fmt = request.response_format
    return fmt is not None and fmt.type != "text"


class Finalizer:
    """restores vault placeholders in released text; json-escaped inside tool arguments and json replies"""

    def __init__(self, policy: EffectivePolicy, request: ChatRequest, vault: GuardVault) -> None:
        self._restorer: Restorer | None = policy.restorer
        self._json = json_mode(request)
        self._vault = vault

    def __call__(self, text: str, kind: SegmentKind) -> str:
        if self._restorer is None or not text:
            return text
        return self._restorer.restore(text, self._vault, json_escape=kind == "tool_args" or self._json)


def report_restores(metrics: GuardMetrics, vault: GuardVault) -> None:
    stats = vault.stats
    for direction, strategy, count in (
        ("restored", "exact", stats.exact),
        ("restored", "case_insensitive", stats.case_insensitive),
        ("unmatched", "none", stats.unmatched),
    ):
        if count:
            metrics.placeholders(direction, strategy, count)


def notable(findings: Iterable[GuardFinding]) -> list[GuardFinding]:
    """output windows produce many allow findings; only detections and errors are kept on the context"""
    return [f for f in findings if f.would_verdict > Verdict.ALLOW or f.error is not None]
