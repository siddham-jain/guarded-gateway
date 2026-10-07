"""eval item schema (C6 §9.2 / C7 §9.2) and loader"""

import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from gg.config.yaml_loader import load_yaml_file
from gg.core.jsonutil import canonical_json, sha256_hex
from gg.core.schema import Role, StrictModel
from gg.guardrails.fakes import FakeValues

type Action = Literal["allow", "flag", "redact", "block"]
ACTIONS: tuple[Action, ...] = ("allow", "flag", "redact", "block")

_VAR_RE = re.compile(r"\{\{([a-z_][a-z0-9_]*)\}\}")


class EvalMessage(StrictModel):
    role: Role
    content: str


class ExpectedRedaction(StrictModel):
    label: str
    text: str


class Expected(StrictModel):
    action: Action
    guardrails_any: tuple[str, ...] = ()
    redactions: tuple[ExpectedRedaction, ...] = ()
    must_not_release: tuple[str, ...] = ()
    released_contains: tuple[str, ...] = ()


class OutputRequest(StrictModel):
    messages: tuple[EvalMessage, ...] = (EvalMessage(role="user", content="hello"),)
    response_format: dict[str, Any] | None = None


class EvalItem(StrictModel):
    id: Annotated[str, Field(pattern=r"^(in|out)-[a-z]+-\d{3}$")]
    stage: Literal["input", "output"]
    category: str
    label: Literal["adversarial", "benign"]
    split: Literal["dev", "heldout"] = "dev"
    pair: str | None = None
    messages: tuple[EvalMessage, ...] = ()
    vars: dict[str, str] = Field(default_factory=dict)
    request: OutputRequest = OutputRequest()
    vault: dict[str, str] = Field(default_factory=dict)
    output: str | None = None
    chunks: tuple[str, ...] | None = None
    expected: Expected
    source_inspiration: str = ""
    notes: str = ""
    added_in: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]

    @model_validator(mode="after")
    def _stage_fields(self) -> Self:
        if self.stage == "input" and not self.messages:
            raise ValueError("input items need messages")
        if self.stage == "output" and self.output is None:
            raise ValueError("output items need output")
        return self

    def expand(self, text: str, fakes: FakeValues) -> str:
        return fakes.expand(_VAR_RE.sub(lambda m: self.vars.get(m.group(1), m.group(0)), text))


_ITEMS = TypeAdapter(list[EvalItem])


def load_items(items_dir: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for path in sorted(items_dir.glob("*.yaml")):
        raw = load_yaml_file(path)
        items.extend(_ITEMS.validate_python(raw or []))
    ids = [i.id for i in items]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate eval item ids: {', '.join(duplicates)}")
    return items


def items_digest(items: list[EvalItem]) -> str:
    return "sha256:" + sha256_hex(canonical_json([i.model_dump(mode="json") for i in items]))
