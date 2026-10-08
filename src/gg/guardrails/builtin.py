from gg.guardrails.input import register_input_guards
from gg.guardrails.ml.guards import register_ml_guards
from gg.guardrails.output import register_output_guards
from gg.guardrails.registry import GuardRegistry, new_registry, register
from gg.guardrails.remote import jev, promptguard


def default_registry() -> GuardRegistry:
    registry = new_registry()
    register_input_guards(registry)
    register_output_guards(registry)
    register_ml_guards(registry)
    register(registry, "jev_injection", jev.JevInjectionCfg, jev.create)
    register(registry, "promptguard", promptguard.PromptGuardCfg, promptguard.create_input)
    register(registry, "promptguard_output", promptguard.PromptGuardCfg, promptguard.create_output)
    return registry
