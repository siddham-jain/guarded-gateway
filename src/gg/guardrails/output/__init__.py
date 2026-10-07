from gg.guardrails.output import json_schema, pii_leak, pii_restore, secrets_out
from gg.guardrails.registry import GuardRegistry, register


def register_output_guards(registry: GuardRegistry) -> None:
    register(registry, "secrets_out", secrets_out.SecretsOutCfg, secrets_out.create)
    register(registry, "pii_leak", pii_leak.PiiLeakCfg, pii_leak.create)
    register(registry, "json_schema", json_schema.JsonSchemaCfg, json_schema.create)
    register(registry, "pii_restore", pii_restore.PiiRestoreCfg, pii_restore.create)
