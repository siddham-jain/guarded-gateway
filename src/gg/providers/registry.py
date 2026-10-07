from gg.plugins.registry import Registry
from gg.providers.anthropic.adapter import AnthropicAdapter
from gg.providers.base import ProviderAdapter
from gg.providers.catalog.catalog import Catalog
from gg.providers.gemini.adapter import GeminiAdapter
from gg.providers.mock.adapter import MockAdapter
from gg.providers.openai_compat.adapter import OpenAICompatibleAdapter
from gg.providers.runtime import AdapterDeps, ProviderRuntime

type ProviderRegistry = Registry[ProviderAdapter, AdapterDeps]


def register(registry: ProviderRegistry) -> None:
    """adapter types keyed by `providers.<name>.type`; native anthropic/gemini adapters register alike"""
    registry.register("openai_compat", OpenAICompatibleAdapter.from_runtime, config_model=ProviderRuntime)
    registry.register("mock", MockAdapter.from_runtime, config_model=ProviderRuntime)
    registry.register("anthropic", AnthropicAdapter.from_runtime, config_model=ProviderRuntime)
    registry.register("gemini", GeminiAdapter.from_runtime, config_model=ProviderRuntime)


def adapter_registry() -> ProviderRegistry:
    registry: ProviderRegistry = Registry("provider")
    register(registry)
    return registry


def build_adapters(
    catalog: Catalog, deps: AdapterDeps, registry: ProviderRegistry | None = None
) -> dict[str, ProviderAdapter]:
    """one adapter per enabled provider; the composition root closes deps.http on shutdown"""
    registry = registry or adapter_registry()
    return {
        name: registry.create(rt.type, rt, deps)
        for name, rt in catalog.providers.items()
        if rt.type in registry
    }
