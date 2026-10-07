import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gg.core.deployment import Deployment
from gg.core.jsonutil import dumps_str
from gg.core.schema import ChatRequest, CompletionTokensDetails, PromptTokensDetails, Usage
from gg.core.usage import UsageRecord, UsageSource

DEFAULT_USAGE_PATHS: Mapping[str, tuple[str, ...]] = {
    "prompt_tokens": ("prompt_tokens", "input_tokens"),
    "completion_tokens": ("completion_tokens", "output_tokens"),
    "cached_tokens": (
        "prompt_tokens_details.cached_tokens",
        "prompt_cache_hit_tokens",
        "cached_tokens",
        "input_tokens_details.cached_tokens",
    ),
    "cache_write_tokens": ("prompt_tokens_details.cache_write_tokens", "cache_creation_input_tokens"),
    "reasoning_tokens": (
        "completion_tokens_details.reasoning_tokens",
        "reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
    ),
}


def get_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            items: list[Any] = current
            current = items[int(part)] if int(part) < len(items) else None
        else:
            return None
    return current


def _first_int(raw: Mapping[str, Any], paths: Sequence[str]) -> int | None:
    for path in paths:
        value = get_path(raw, path)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return None


@dataclass(frozen=True, slots=True)
class TokenCounts:
    input: int
    output: int
    cached: int = 0
    cache_write: int = 0
    reasoning: int = 0

    def clamped(self) -> "TokenCounts":
        # invariants the cost math relies on: cached + write <= input, reasoning <= output
        cached = min(self.cached, self.input)
        write = min(self.cache_write, self.input - cached)
        return TokenCounts(self.input, self.output, cached, write, min(self.reasoning, self.output))

    def to_usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.input,
            completion_tokens=self.output,
            total_tokens=self.input + self.output,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=self.cached, cache_write_tokens=self.cache_write
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
        return UsageRecord(
            provider=dep.provider,
            deployment_id=dep.id,
            upstream_model=dep.upstream_model,
            input_tokens=self.input,
            output_tokens=self.output,
            cached_input_tokens=self.cached,
            cache_write_tokens=self.cache_write,
            reasoning_tokens=self.reasoning,
            usage_source=source,
            service_tier=service_tier,
            raw=dict(raw or {}),
        )


def counts_from_usage(
    raw: Mapping[str, Any], overrides: Mapping[str, Sequence[str]] | None = None
) -> TokenCounts | None:
    """normalises an openai-shaped (or vendor variant) usage object; None when no token counts are present"""
    paths = {**DEFAULT_USAGE_PATHS, **(overrides or {})}
    prompt = _first_int(raw, paths["prompt_tokens"])
    completion = _first_int(raw, paths["completion_tokens"])
    if prompt is None and completion is None:
        return None
    return TokenCounts(
        input=prompt or 0,
        output=completion or 0,
        cached=_first_int(raw, paths["cached_tokens"]) or 0,
        cache_write=_first_int(raw, paths["cache_write_tokens"]) or 0,
        reasoning=_first_int(raw, paths["reasoning_tokens"]) or 0,
    ).clamped()


@functools.cache
def _encoder() -> Callable[[str], int] | None:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("o200k_base")
    except Exception:
        # offline or missing encoding file: fall back to a character heuristic
        return None
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    encode = _encoder()
    return encode(text) if encode is not None else max(1, (len(text) + 3) // 4)


def estimate_prompt_tokens(request: ChatRequest) -> int:
    parts: list[str] = []
    for message in request.messages:
        parts.append(message.text())
        for call in message.tool_calls or ():
            parts.append(call.function.name)
            parts.append(call.function.arguments)
    if request.tools:
        parts.append(dumps_str([t.model_dump(mode="json", exclude_none=True) for t in request.tools]))
    # ~4 tokens of chat-template overhead per message
    return estimate_text_tokens("\n".join(parts)) + 4 * len(request.messages)
