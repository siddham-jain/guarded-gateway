from gg.guardrails.input import injection_rules, normalize, pii_regex, secrets
from gg.guardrails.registry import GuardRegistry, register


def register_input_guards(registry: GuardRegistry) -> None:
    register(registry, "normalize", normalize.NormalizeCfg, normalize.create)
    register(registry, "injection_rules", injection_rules.InjectionRulesCfg, injection_rules.create)
    register(registry, "secrets", secrets.SecretsCfg, secrets.create)
    register(registry, "pii_regex", pii_regex.PiiRegexCfg, pii_regex.create)
