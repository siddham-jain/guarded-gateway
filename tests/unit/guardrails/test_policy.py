from pathlib import Path
from typing import Any

import pytest

from gg.config.loader import ConfigError
from gg.core.clock import SystemClock
from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import GuardContext, GuardFinding, GuardStage, Mode, Streaming, finding
from gg.guardrails.builtin import default_registry
from gg.guardrails.errors import GuardrailOverrideRejectedError
from gg.guardrails.policy.effective import PolicyError
from gg.guardrails.policy.loader import apply_patch
from gg.guardrails.registry import GuardDeps, register
from gg.guardrails.setup import build_guardrails
from tests.conftest import make_key, make_request
from tests.unit.guardrails.support import DEFAULT_POLICY, POLICY_DIR, policy_doc, policy_set, write_policy


def test_default_policy_loads_with_both_chains() -> None:
    policies = policy_set(key_policy_ids=["default"])
    eff = policies.effective(make_key(), make_request())
    assert [g.name for g in eff.input_chain.guards] == [
        "normalize",
        "injection_rules",
        "secrets",
        "pii_regex",
        "pii_ner",
        "topic",
    ]
    assert [g.name for g in eff.output_chain.guards] == [
        "secrets_out",
        "pii_leak",
        "json_schema",
        "toxicity",
        "pii_restore",
    ]
    assert eff.restorer is not None
    assert [g.name for g in eff.output_detectors(streaming=True).guards] == [
        "secrets_out",
        "pii_leak",
        "toxicity",
    ]
    assert eff.ref.header() == f"default@1.3.0+{eff.hash}"
    assert len(eff.hash) == 12


def test_effective_policies_are_cached_and_hash_is_stable() -> None:
    policies = policy_set()
    a = policies.effective(make_key(), make_request())
    assert policies.effective(make_key(id="other-key"), make_request()) is a
    assert policy_set().effective(make_key(), make_request()).hash == a.hash
    assert len(policies.hash) == 12


def test_hash_ignores_formatting_but_tracks_values(tmp_path: Path) -> None:
    base = policy_set().effective(make_key(), make_request()).hash
    reformatted = write_policy(tmp_path / "a", policy_doc())
    assert policy_set(reformatted).effective(make_key(), make_request()).hash == base
    doc = policy_doc()
    doc["output"]["streaming"]["window_chars"] = 300
    changed = write_policy(tmp_path / "b", doc)
    assert policy_set(changed).effective(make_key(), make_request()).hash != base


def test_key_tag_override_resolves_a_different_effective_policy() -> None:
    policies = policy_set()
    plain = policies.effective(make_key(), make_request())
    dev = policies.effective(make_key(tags=["internal-dev"]), make_request())
    rules = dev.input_chain.get("injection_rules")
    assert rules is not None
    assert rules.settings.mode is Mode.SHADOW
    assert dev.hash != plain.hash
    strict = policies.effective(make_key(tags=["strict"]), make_request())
    assert strict.doc.output.streaming.mode == "buffer"


def test_override_matches_on_model_and_key_id(tmp_path: Path) -> None:
    doc = policy_doc(
        overrides=[
            {
                "name": "only-mock",
                "match": {"models": ["mock/*"], "key_ids": ["special"]},
                "patch": {"output.streaming.window_chars": 64},
            }
        ]
    )
    policies = policy_set(write_policy(tmp_path, doc))
    hit = policies.effective(make_key(id="special"), make_request(model="mock/echo"))
    miss = policies.effective(make_key(id="special"), make_request(model="gg/auto"))
    assert hit.doc.output.streaming.window_chars == 64
    assert miss.doc.output.streaming.window_chars == 200


def test_apply_patch_paths() -> None:
    raw: dict[str, Any] = policy_doc()
    apply_patch(raw, "input.guards.pii_regex.entities.US_SSN", "redact")
    apply_patch(raw, "input.guards.secrets.mode", "enforce")
    apply_patch(raw, "output.streaming.abort", "error_frame")
    assert raw["input"]["guards"][3]["entities"]["US_SSN"] == "redact"
    assert raw["output"]["streaming"]["abort"] == "error_frame"
    with pytest.raises(PolicyError, match="no guard named"):
        apply_patch(raw, "input.guards.nope.mode", "off")
    with pytest.raises(PolicyError, match="start with"):
        apply_patch(raw, "routing.threshold", 1)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"input.guards.secrets.typo_field": 1}, "typo_field"),
        ({"input.guards.secrets.mode": "shadow"}, "allow_unredacted_upstream"),
        ({"output.streaming.overlap_chars": 999}, "overlap_chars"),
    ],
)
def test_bad_overrides_fail_at_startup(tmp_path: Path, patch: dict[str, Any], message: str) -> None:
    doc = policy_doc(overrides=[{"name": "bad", "match": {"key_tags_any": ["x"]}, "patch": patch}])
    with pytest.raises(ConfigError, match=message):
        policy_set(write_policy(tmp_path, doc))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"version": "1.0"}, "version"),
        ({"input": {"guards": [{"guard": "normalize"}, {"guard": "normalize"}]}}, "duplicate input guard"),
        ({"input": {"guards": [{"guard": "no_such_guard"}]}}, "unknown guardrail 'no_such_guard'"),
        ({"input": {"guards": [{"guard": "secrets_out", "pack": "rules/secrets.v1.yaml"}]}}, "output guard"),
        ({"input": {"guards": [{"guard": "injection_rules", "pack": "rules/missing.yaml"}]}}, "not found"),
        ({"defaults": {"mode": "loud"}}, "defaults.mode"),
        ({"surprise": True}, "surprise"),
    ],
)
def test_invalid_policies_name_the_problem(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        policy_set(write_policy(tmp_path, policy_doc(**change)))


def test_shadow_redaction_needs_explicit_opt_in(tmp_path: Path) -> None:
    doc = policy_doc()
    doc["input"]["guards"][3]["mode"] = "shadow"
    with pytest.raises(ConfigError, match="allow_unredacted_upstream"):
        policy_set(write_policy(tmp_path / "a", doc))
    doc["input"]["allow_unredacted_upstream"] = True
    policy_set(write_policy(tmp_path / "b", doc))


def test_unknown_key_policy_id_fails_startup() -> None:
    with pytest.raises(ConfigError, match="unknown guardrail policy 'strict-v9'"):
        policy_set(key_policy_ids=["default", "strict-v9"])


def test_missing_policy_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no guardrail policy files"):
        policy_set(tmp_path)


def _ext(**fields: Any) -> Any:
    return make_request(gg={"guardrails": fields})


def test_request_tightening_enables_and_enforces(tmp_path: Path) -> None:
    doc = policy_doc()
    doc["input"]["guards"][1]["mode"] = "shadow"
    doc["output"]["guards"][2]["mode"] = "off"
    policies = policy_set(write_policy(tmp_path, doc))
    base = policies.effective(make_key(), make_request())
    assert base.output_chain.get("json_schema") is None
    tightened = policies.effective(make_key(), _ext(enable=["json_schema"], enforce_shadowed=True))
    json_guard = tightened.output_chain.get("json_schema")
    rules = tightened.input_chain.get("injection_rules")
    assert json_guard is not None
    assert json_guard.settings.mode is Mode.ENFORCE
    assert rules is not None
    assert rules.settings.mode is Mode.ENFORCE
    buffered = policies.effective(make_key(), _ext(output_stream_mode="buffer"))
    assert buffered.doc.output.streaming.mode == "buffer"
    assert len({base.hash, tightened.hash, buffered.hash}) == 3


def test_request_tightening_rejects_unknown_guards_and_disallowed_policies(tmp_path: Path) -> None:
    with pytest.raises(GuardrailOverrideRejectedError) as info:
        policy_set().effective(make_key(), _ext(enable=["telepathy"]))
    assert info.value.status == 400
    assert info.value.code == "guardrail_override_rejected"
    assert info.value.param == "gg.guardrails"
    locked = policy_set(write_policy(tmp_path, policy_doc(tightening={"allow_request": False})))
    with pytest.raises(GuardrailOverrideRejectedError):
        locked.effective(make_key(), _ext(enforce_shadowed=True))


class EchoCfg(StrictModel):
    threshold: float = 0.5


class FakeClassifier:
    """stands in for a part-2 ml guard: registers by name, configured from the policy"""

    name: str = "echo_classifier"
    stage: GuardStage = "input"
    tier: int = 2
    streaming: Streaming = "windowed"

    def __init__(self, cfg: EchoCfg) -> None:
        self.cfg = cfg

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        return finding(self.name, "input", Verdict.FLAG, score=self.cfg.threshold)


def test_new_guards_plug_in_through_the_registry(tmp_path: Path) -> None:
    registry = default_registry()

    def create(cfg: EchoCfg, deps: GuardDeps) -> FakeClassifier:
        return FakeClassifier(cfg)

    register(registry, "echo_classifier", EchoCfg, create)
    doc = policy_doc()
    doc["input"]["guards"].append({"guard": "echo_classifier", "threshold": 0.9, "on_error": "block"})
    policies = policy_set(write_policy(tmp_path, doc), registry=registry)
    eff = policies.effective(make_key(), make_request())
    bound = eff.input_chain.get("echo_classifier")
    assert bound is not None
    assert isinstance(bound.guard, FakeClassifier)
    assert bound.guard.cfg.threshold == 0.9
    assert bound.tier == 2


def test_default_yaml_is_the_documented_shape() -> None:
    assert DEFAULT_POLICY["id"] == "default"
    assert {g["guard"] for g in DEFAULT_POLICY["input"]["guards"]} == {
        "normalize",
        "injection_rules",
        "secrets",
        "pii_regex",
        "pii_ner",
        "promptguard",
        "topic",
    }


def test_build_guardrails_resolves_configured_keys_up_front() -> None:
    keys = [make_key(), make_key(id="dev-key", tags=["internal-dev"])]
    guardrails = build_guardrails(POLICY_DIR, clock=SystemClock(), keys=keys)
    assert (
        guardrails.policies.effective(keys[1], make_request()).input_chain.get("injection_rules") is not None
    )
    with pytest.raises(ConfigError, match="unknown guardrail policy"):
        build_guardrails(POLICY_DIR, clock=SystemClock(), keys=[make_key(guardrails={"policy_id": "nope"})])
