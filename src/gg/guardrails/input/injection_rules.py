"""tier 1: the versioned injection rule pack; only rules tagged in block_tags may block, the rest flag"""

from gg.core.guard_types import Verdict
from gg.core.schema import Role, StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Segment, Streaming, finding
from gg.guardrails.registry import GuardDeps
from gg.guardrails.rules import CompiledInjectionRule, InjectionPack


class InjectionRulesCfg(StrictModel):
    pack: str
    roles: tuple[Role, ...] = ("user", "assistant", "tool")
    block_tags: tuple[str, ...] = ()


def _hits(rule: CompiledInjectionRule, seg: Segment) -> bool:
    if "inspect" in rule.views and rule.regex.search(seg.view):
        return True
    return "decoded" in rule.views and any(rule.regex.search(d.text) for d in seg.decoded)


class InjectionRules:
    name: str = "injection_rules"
    stage: GuardStage = "input"
    tier: int = 1
    streaming: Streaming = "windowed"

    def __init__(self, cfg: InjectionRulesCfg, pack: InjectionPack) -> None:
        self._roles = frozenset(cfg.roles)
        self._block_tags = frozenset(cfg.block_tags)
        self._pack = pack

    def matches(self, segments: tuple[Segment, ...]) -> list[CompiledInjectionRule]:
        scoped = [s for s in segments if s.role in self._roles and s.kind != "tool_args"]
        return [rule for rule in self._pack.rules if any(_hits(rule, seg) for seg in scoped)]

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        hits = self.matches(gctx.segments)
        if not hits:
            return finding(self.name, "input")
        blocking = [r for r in hits if r.tags & self._block_tags]
        top = (blocking or hits)[0]
        tags = sorted({t for r in hits for t in r.tags})
        return finding(
            self.name,
            "input",
            Verdict.BLOCK if blocking else Verdict.FLAG,
            score=1.0,
            reason=top.id,
            labels=tuple(tags),
        )


def create(cfg: InjectionRulesCfg, deps: GuardDeps) -> InjectionRules:
    return InjectionRules(cfg, deps.packs.injection(cfg.pack))
