from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from gg.core.deployment import Deployment
from gg.core.schema import CompletionTokensDetails, PromptTokensDetails, Usage
from gg.core.usage import UsageRecord, UsageSource
from gg.providers.usage import TokenCounts, get_path


@dataclass(frozen=True, slots=True)
class AnthropicTokenCounts(TokenCounts):
    """token counts plus 1h cache writes, which anthropic prices separately from 5m writes"""

    cache_write_1h: int = 0

    def clamped(self) -> "AnthropicTokenCounts":
        base = TokenCounts.clamped(self)
        write_1h = min(self.cache_write_1h, base.input - base.cached - base.cache_write)
        return AnthropicTokenCounts(
            base.input, base.output, base.cached, base.cache_write, base.reasoning, write_1h
        )

    def to_usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.input,
            completion_tokens=self.output,
            total_tokens=self.input + self.output,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=self.cached, cache_write_tokens=self.cache_write + self.cache_write_1h
            ),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=self.reasoning),
        )

    def to_record(
        self,
        dep: Deployment,
        *,
        source: UsageSource,
        raw: Mapping[str, Any] | None = None,
        service_tier: str | None = None,
    ) -> UsageRecord:
        record = TokenCounts.to_record(self, dep, source=source, raw=raw, service_tier=service_tier)
        tier = raw.get("service_tier") if raw else None
        return replace(
            record,
            cache_write_1h_tokens=self.cache_write_1h,
            service_tier=service_tier or (tier if isinstance(tier, str) else None),
        )


def _int(raw: Mapping[str, Any], path: str) -> int | None:
    value = get_path(raw, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0, int(value))


def counts_from_usage(raw: Mapping[str, Any]) -> AnthropicTokenCounts | None:
    """anthropic usage -> openai semantics: input counts every prompt token incl. cache reads and writes"""
    uncached = _int(raw, "input_tokens")
    output = _int(raw, "output_tokens")
    if uncached is None and output is None:
        return None
    read = _int(raw, "cache_read_input_tokens") or 0
    created = _int(raw, "cache_creation_input_tokens") or 0
    write_5m = _int(raw, "cache_creation.ephemeral_5m_input_tokens")
    write_1h = _int(raw, "cache_creation.ephemeral_1h_input_tokens") or 0
    if write_5m is None:
        write_5m = max(0, created - write_1h)
    return AnthropicTokenCounts(
        input=(uncached or 0) + max(created, write_5m + write_1h) + read,
        output=output or 0,
        cached=read,
        cache_write=write_5m,
        reasoning=_int(raw, "output_tokens_details.thinking_tokens") or 0,
        cache_write_1h=write_1h,
    ).clamped()
