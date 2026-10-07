import pytest

from gg.guardrails.fakes import KINDS, FakeValues, fake
from gg.guardrails.pii_patterns import iban_ok, luhn_ok
from gg.guardrails.rules import InjectionRule, RulePackError, RulePackStore, SecretRule
from gg.guardrails.secrets_rules import EntropyCfg, SecretDetector
from tests.unit.guardrails.support import POLICY_DIR

PACKS = RulePackStore(POLICY_DIR)
INJECTION = PACKS.injection("rules/injection.v1.yaml")
SECRETS = PACKS.secrets("rules/secrets.v1.yaml")
FAKES = FakeValues()


@pytest.mark.parametrize("rule", INJECTION.doc.rules, ids=lambda r: r.id)
def test_injection_rule_examples(rule: InjectionRule) -> None:
    compiled = next(r for r in INJECTION.rules if r.id == rule.id)
    for text in rule.examples.match:
        assert compiled.regex.search(text), f"{compiled.id} should match: {text}"
    for text in rule.examples.no_match:
        assert not compiled.regex.search(text), f"{compiled.id} should not match: {text}"


@pytest.mark.parametrize("rule", SECRETS.doc.rules, ids=lambda r: r.id)
def test_secret_rule_examples(rule: SecretRule) -> None:
    detector = SecretDetector(SECRETS, EntropyCfg())
    for template in rule.examples.match:
        text = FAKES.expand(template)
        assert rule.id in {m.rule for m in detector.find(text)}, text
    for template in rule.examples.no_match:
        text = FAKES.expand(template)
        assert detector.find(text) == [], text


def test_packs_have_enough_rules_and_unique_ids() -> None:
    assert len(INJECTION.rules) >= 25
    assert len(SECRETS.rules) >= 15
    assert len({r.id for r in INJECTION.rules}) == len(INJECTION.rules)
    assert len({r.id for r in SECRETS.rules}) == len(SECRETS.rules)
    assert PACKS.digest("rules/injection.v1.yaml") == INJECTION.digest


def test_packs_cannot_escape_the_policy_dir() -> None:
    with pytest.raises(RulePackError):
        PACKS.injection("../keys.yaml")


def test_fakes_are_deterministic_and_valid() -> None:
    assert FakeValues(1).get("github_pat") == FakeValues(1).get("github_pat")
    assert FakeValues(1).get("github_pat") != FakeValues(2).get("github_pat")
    assert luhn_ok(fake("card"))
    assert iban_ok(fake("iban"))
    assert FAKES.expand("{{fake:email[:4]}}{{fake:email[4:]}}") == FAKES.get("email")
    for kind in KINDS:
        assert FAKES.get(kind)
    with pytest.raises(KeyError):
        FAKES.get("nope")
