"""registry entries of the model-backed guards"""

from gg.guardrails.ml import grounding, pii_ner, topic, toxicity
from gg.guardrails.registry import GuardRegistry, register


def register_ml_guards(registry: GuardRegistry) -> None:
    register(registry, "pii_ner", pii_ner.PiiNerCfg, pii_ner.create)
    register(registry, "topic", topic.TopicCfg, topic.create)
    register(registry, "toxicity", toxicity.ToxicityCfg, toxicity.create)
    register(registry, "grounding", grounding.GroundingCfg, grounding.create)
