"""finalizer: puts this request's own vault values back into released output"""

from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.registry import GuardDeps
from gg.guardrails.vault import GuardVault


class PiiRestoreCfg(StrictModel):
    pass


class PiiRestore:
    name: str = "pii_restore"
    stage: GuardStage = "output"
    tier: int = 9
    streaming: Streaming = "windowed"

    def restore(self, text: str, vault: GuardVault, /, *, json_escape: bool = False) -> str:
        return vault.restore(text, json_escape=json_escape)

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        return finding(self.name, "output")


def create(cfg: PiiRestoreCfg, deps: GuardDeps) -> PiiRestore:
    return PiiRestore()
