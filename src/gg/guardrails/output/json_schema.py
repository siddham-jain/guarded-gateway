"""buffer guard: output must be json (json_object) or match the request's schema (json_schema).

strict validation only; no repair library is installed, so an invalid reply is flagged (or blocked by policy).
"""

import json
from typing import Any, Literal

from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.output.jsonschema_lite import validate
from gg.guardrails.registry import GuardDeps


class JsonSchemaCfg(StrictModel):
    on_invalid: Literal["flag", "block"] = "flag"


class JsonSchemaGuard:
    name: str = "json_schema"
    stage: GuardStage = "output"
    tier: int = 2
    streaming: Streaming = "buffer"

    def __init__(self, cfg: JsonSchemaCfg) -> None:
        self._verdict = Verdict.BLOCK if cfg.on_invalid == "block" else Verdict.FLAG

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        fmt = gctx.request.response_format
        content = [s for s in gctx.segments if s.kind == "content"]
        if fmt is None or fmt.type == "text" or not content:
            return finding(self.name, "output")
        try:
            value: Any = json.loads("".join(s.text for s in content))
        except ValueError:
            return finding(
                self.name, "output", self._verdict, reason="invalid_json", labels=("invalid_json",)
            )
        schema = fmt.json_schema.schema_ if fmt.json_schema is not None else None
        if fmt.type == "json_schema" and schema is not None and validate(value, schema):
            return finding(
                self.name, "output", self._verdict, reason="schema_mismatch", labels=("schema_mismatch",)
            )
        return finding(self.name, "output")


def create(cfg: JsonSchemaCfg, deps: GuardDeps) -> JsonSchemaGuard:
    return JsonSchemaGuard(cfg)
