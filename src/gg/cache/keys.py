"""exact-cache key over the normalised, redacted request (C8 §4.1); pure"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from gg.cache.base import AliasRevision, CacheKey, SemanticTags
from gg.core.context import RequestContext
from gg.core.jsonutil import canonical_json, sha256_hex
from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest, ContentPart, ImagePart, Message, TextPart
from gg.guardrails.base import POLICY_REF

NAMESPACE = "gg:c:v1"

# never part of the key: transport, attribution and the cache controls themselves
_EXCLUDED_PARAMS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "user",
        "metadata",
        "store",
        "service_tier",
        "prompt_cache_key",
        "safety_identifier",
        "gg",
    }
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SPACE_RE = re.compile(r"\s+")


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list | tuple):
        return [_drop_none(v) for v in value]
    return value


def _part(part: ContentPart) -> Any:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        return {"type": "image", "sha256": sha256_hex(part.image_url.url), "detail": part.image_url.detail}
    return part.model_dump(mode="json", by_alias=True)


def _content(content: str | tuple[ContentPart, ...] | None) -> Any:
    if content is None or isinstance(content, str):
        return content
    parts = [_part(p) for p in content]
    # a plain string and a single text part are the same request
    if len(parts) == 1 and isinstance(content[0], TextPart):
        return content[0].text
    return parts


def canonical_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": _content(message.content)}
    if message.name is not None:
        out["name"] = message.name
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        out["tool_calls"] = [c.model_dump(mode="json", by_alias=True) for c in message.tool_calls]
    if message.refusal is not None:
        out["refusal"] = message.refusal
    return out


def canonical_params(request: ChatRequest) -> dict[str, Any]:
    """every non-message parameter, unknown extras included; only n=1, empty lists and logprobs=false elide"""
    raw = request.model_dump(mode="json", by_alias=True, exclude_none=True, exclude=set(_EXCLUDED_PARAMS))
    if raw.get("n") == 1:
        del raw["n"]
    for name in ("tools", "stop"):
        if raw.get(name) == []:
            del raw[name]
    if raw.get("logprobs") is False:
        del raw["logprobs"]
    if request.gg is not None and request.gg.route_threshold is not None:
        raw["route_threshold"] = request.gg.route_threshold
    return raw


def semantic_text(request: ChatRequest) -> str:
    """the single user message, nfkc-normalised with whitespace collapsed"""
    text = request.last_user_text() or ""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def number_signature(text: str) -> str:
    return _short(canonical_json(sorted(_NUMBER_RE.findall(text))))


def scope_of(key: KeyPolicy) -> str:
    return "g" if key.cache.scope == "global" else f"k:{key.id}"


def policy_sha(ctx: RequestContext) -> str:
    ref = ctx.get(POLICY_REF)
    return ref.hash if ref is not None else "none"


def default_alias_revision(ctx: RequestContext) -> str:
    """config_hash covers models, routing and the effective policy; key routing overrides move alpha"""
    routing = ctx.key.routing.model_dump(mode="json")
    return _short(
        canonical_json({"alias": ctx.original.model, "config": ctx.config_hash, "routing": routing})
    )


def _short(data: bytes | str) -> str:
    return sha256_hex(data)[:16]


def _system(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m["role"] in ("system", "developer")]


class CacheKeyBuilder:
    def __init__(self, alias_revision: AliasRevision | None = None) -> None:
        self._alias_revision = alias_revision or default_alias_revision

    def build(self, ctx: RequestContext) -> CacheKey:
        source = ctx.scrubbed or ctx.request
        scope = scope_of(ctx.key)
        alias = ctx.original.model
        alias_rev = self._alias_revision(ctx)
        policy = policy_sha(ctx)
        messages = [canonical_message(m) for m in source.messages]
        params = canonical_params(source)
        payload = _drop_none(
            {"alias": alias, "alias_rev": alias_rev, "policy": policy, "messages": messages, "params": params}
        )
        payload_sha = sha256_hex(canonical_json(payload))
        tags = SemanticTags(
            scope=_short(scope),
            alias=_short(alias),
            alias_rev=_short(alias_rev),
            policy_sha=_short(policy),
            system_sha=_short(canonical_json(_drop_none(_system(messages)))),
            params_sha=_short(canonical_json(_drop_none(params))),
            num_sig=number_signature(source.last_user_text() or ""),
        )
        return CacheKey(
            redis_key=f"{NAMESPACE}:{scope}:{payload_sha}", scope=scope, payload_sha=payload_sha, tags=tags
        )
