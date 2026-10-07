from pydantic import Field

from gg.core.schema import StrictModel


class CachePolicy(StrictModel):
    # automatic top-level cache_control once the prompt clears the model minimum (caps.cache_min_tokens)
    enabled: bool = True
    min_static_tokens: int = Field(default=2048, ge=0)


class AnthropicQuirks(StrictModel):
    """provider-level knobs for the native messages adapter (`providers.anthropic.quirks`)"""

    api_version: str = "2023-06-01"
    expose_reasoning: bool = False
    thinking_budgets: dict[str, int] = Field(
        default_factory=lambda: {"low": 1024, "medium": 4096, "high": 16000}
    )
    cache: CachePolicy = Field(default_factory=CachePolicy)
