from typing import Any, Literal

import pytest

from gg.core.guard_types import Verdict
from gg.core.schema import ChatRequest
from gg.guardrails.base import GuardContext, Segment
from gg.guardrails.fakes import FakeValues
from gg.guardrails.output.json_schema import JsonSchemaCfg, JsonSchemaGuard
from gg.guardrails.output.jsonschema_lite import validate
from gg.guardrails.output.pii_leak import PiiLeak, PiiLeakCfg
from gg.guardrails.output.pii_restore import PiiRestore
from gg.guardrails.output.secrets_out import SecretsOut, SecretsOutCfg
from gg.guardrails.rules import RulePackStore
from gg.guardrails.secrets_rules import EntropyCfg, SecretDetector
from gg.guardrails.vault import GuardVault
from tests.unit.guardrails.support import POLICY_DIR

FAKES = FakeValues()
PACKS = RulePackStore(POLICY_DIR)


def out_ctx(
    text: str, *, vault: GuardVault | None = None, final: bool = True, **request: Any
) -> GuardContext:
    req = ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "q"}], **request}
    )
    return GuardContext(
        stage="output",
        request_id="r",
        segments=(Segment(index=0, role="assistant", kind="content", msg=-1, text=text),),
        request=req,
        vault=vault or GuardVault(),
        is_final=final,
    )


def secrets_out(action: Literal["redact", "block"] = "redact") -> SecretsOut:
    cfg = SecretsOutCfg(pack="rules/secrets.v1.yaml", action=action)
    return SecretsOut(cfg, SecretDetector(PACKS.secrets(cfg.pack), EntropyCfg()))


async def test_secrets_out_redacts_and_reports_continuations() -> None:
    key = FAKES.get("openai_key")
    f = await secrets_out().check(out_ctx(f"use {key} now"))
    assert f.verdict is Verdict.REDACT
    (span,) = f.redactions
    assert span.continuation is not None
    pem = FAKES.get("pem_private_key")
    f = await secrets_out().check(out_ctx(f"key:\n{pem[:80]}"))
    (span,) = f.redactions
    assert span.end == len(f"key:\n{pem[:80]}")
    assert "END" in (span.continuation or "")
    blocked = await secrets_out("block").check(out_ctx(f"use {key}"))
    assert blocked.verdict is Verdict.BLOCK


@pytest.mark.parametrize(
    ("text", "held"),
    [
        ("all done.", 5),
        ("the key is sk-pr", 5),
        ("text then -----BEGIN RSA PRIV", 19),
        ("ends with space ", 0),
    ],
)
def test_secrets_out_holdback(text: str, held: int) -> None:
    assert secrets_out().holdback(text) == held


async def test_pii_leak_ignores_this_requests_own_values_and_allow_values() -> None:
    vault = GuardVault()
    vault.add("EMAIL", "dana@corp.example")
    guard = PiiLeak(PiiLeakCfg(allow_values=("user@example.com",)))
    text = "dana@corp.example, user@example.com, eve@corp.example"
    f = await guard.check(out_ctx(text, vault=vault))
    assert [(r.label, text[r.start : r.end]) for r in f.redactions] == [("EMAIL", "eve@corp.example")]
    clean = await guard.check(out_ctx("contact [EMAIL_1]", vault=vault))
    assert clean.verdict is Verdict.ALLOW


@pytest.mark.parametrize(("text", "held"), [("call 555 01", 6), ("mail bob@exa", 7), ("done ", 0)])
def test_pii_leak_holdback(text: str, held: int) -> None:
    assert PiiLeak(PiiLeakCfg()).holdback(text) == held


def test_pii_restore_round_trip() -> None:
    vault = GuardVault()
    vault.add("EMAIL", "a@x.com")
    restore = PiiRestore()
    assert restore.restore("to [EMAIL_1]", vault) == "to a@x.com"
    assert restore.restore('{"to": "[email_1]"}', vault, json_escape=True) == '{"to": "a@x.com"}'


SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string", "minLength": 1}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    ("content", "fmt", "verdict"),
    [
        (
            '{"name": "a", "age": 3}',
            {"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}},
            Verdict.ALLOW,
        ),
        ('{"age": 3}', {"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}}, Verdict.FLAG),
        ("```json\n{}\n```", {"type": "json_object"}, Verdict.FLAG),
        ("[1, 2]", {"type": "json_object"}, Verdict.ALLOW),
        ("plain prose", None, Verdict.ALLOW),
    ],
)
async def test_json_schema_guard(content: str, fmt: dict[str, Any] | None, verdict: Verdict) -> None:
    request = {"response_format": fmt} if fmt else {}
    f = await JsonSchemaGuard(JsonSchemaCfg()).check(out_ctx(content, **request))
    assert f.verdict is verdict


async def test_json_schema_can_block() -> None:
    f = await JsonSchemaGuard(JsonSchemaCfg(on_invalid="block")).check(
        out_ctx("nope", response_format={"type": "json_object"})
    )
    assert f.verdict is Verdict.BLOCK


@pytest.mark.parametrize(
    ("value", "schema", "ok"),
    [
        ({"name": "x"}, SCHEMA, True),
        ({"name": ""}, SCHEMA, False),
        ({"name": "x", "extra": 1}, SCHEMA, False),
        ({"name": "x", "age": -1}, SCHEMA, False),
        ({"name": "x", "age": True}, SCHEMA, False),
        ([1, "a"], {"type": "array", "items": {"type": "integer"}}, False),
        ([1, 2], {"type": "array", "items": {"type": "integer"}, "maxItems": 1}, False),
        ("b", {"enum": ["a", "b"]}, True),
        (3, {"anyOf": [{"type": "string"}, {"type": "integer"}]}, True),
        (3, {"oneOf": [{"type": "number"}, {"type": "integer"}]}, False),
        (None, {"type": ["string", "null"]}, True),
        (2.0, {"type": "integer"}, True),
    ],
)
def test_jsonschema_subset(value: Any, schema: dict[str, Any], ok: bool) -> None:
    assert (validate(value, schema) == []) is ok
