"""public cache contracts; the only module other feature packages may import"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from gg.core.context import RequestContext
from gg.core.schema import StrictModel, Usage
from gg.core.usage import UsageRecord

type Layer = Literal["exact", "semantic"]
type LookupResult = Literal["hit", "miss", "bypass", "error", "timeout"]
type StoreResult = Literal["stored", "skipped", "rejected", "error"]
type PutResult = Literal["stored", "exists", "too_large", "error"]
type SingleFlightOutcome = Literal["leader", "waited_hit", "waited_miss", "waited_timeout"]
type DistanceResult = Literal["hit", "miss"]


@dataclass(frozen=True, slots=True)
class SemanticTags:
    """hex digests only, so redis TAG queries need no escaping"""

    scope: str
    alias: str
    alias_rev: str
    policy_sha: str
    system_sha: str
    params_sha: str
    num_sig: str

    def as_dict(self) -> dict[str, str]:
        return {
            "scope": self.scope,
            "alias": self.alias,
            "alias_rev": self.alias_rev,
            "policy_sha": self.policy_sha,
            "system_sha": self.system_sha,
            "params_sha": self.params_sha,
            "num_sig": self.num_sig,
        }


@dataclass(frozen=True, slots=True)
class CacheKey:
    redis_key: str
    scope: str
    payload_sha: str
    tags: SemanticTags

    @property
    def lock_key(self) -> str:
        return f"{self.redis_key}:lk"


class SavedUsage(StrictModel):
    """the original upstream usage, so a hit can price what it saved"""

    provider: str
    deployment_id: str
    upstream_model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def of(cls, record: UsageRecord) -> "SavedUsage":
        return cls(
            provider=record.provider,
            deployment_id=record.deployment_id,
            upstream_model=record.upstream_model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
            reasoning_tokens=record.reasoning_tokens,
        )

    def record(self) -> UsageRecord:
        return UsageRecord(
            provider=self.provider,
            deployment_id=self.deployment_id,
            upstream_model=self.upstream_model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )


class CachedResponse(StrictModel):
    """a guarded reply in placeholder form (before pii restore); never holds a vault value"""

    v: Literal[1] = 1
    content: str | None = None
    refusal: str | None = None
    finish_reason: Literal["stop", "length"]
    usage: Usage | None = None
    upstream: SavedUsage | None = None
    provider: str | None = None
    served_by: str | None = None
    response_model: str
    system_fingerprint: str | None = None
    cost_usd: float | None = None
    created_at: float
    request_id: str
    # writer-vault placeholders present in the text; each must resolve in the reader's vault
    placeholders: tuple[str, ...] = ()


class ResponseCache(Protocol):
    @property
    def available(self) -> bool: ...

    async def get(self, key: str, /) -> CachedResponse | None: ...

    async def put(
        self, key: str, value: CachedResponse, ttl_s: int, /, *, replace: bool = False
    ) -> PutResult:
        """SET NX by default; replace=True overwrites (a client refresh)"""
        ...


class SingleFlight(Protocol):
    async def acquire(self, lock_key: str, ttl_ms: int, /) -> str | None:
        """a release token when this caller leads, None when another request holds the lock"""
        ...

    async def release(self, lock_key: str, token: str, /) -> None: ...

    async def held(self, lock_key: str, /) -> bool: ...


class CacheBackend(ResponseCache, SingleFlight, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    exact_key: str
    distance: float


class SemanticIndex(Protocol):
    @property
    def available(self) -> bool: ...

    async def search(self, vector: Sequence[float], tags: SemanticTags, /) -> SemanticMatch | None:
        """the nearest entry with identical tags, whatever its distance"""
        ...

    async def add(
        self, vector: Sequence[float], tags: SemanticTags, exact_key: str, ttl_s: int, /
    ) -> None: ...


@runtime_checkable
class Embedder(Protocol):
    """unit-length float vectors; one instance is shared with the topic guard"""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str], /) -> list[list[float]]: ...


class CacheHooks(Protocol):
    """C10 catalogue #16, #17, #21-26; gg.observability may implement it structurally"""

    def lookup(self, layer: Layer, result: LookupResult, /) -> None: ...
    def lookup_duration(self, layer: Layer, seconds: float, /) -> None: ...
    def semantic_distance(self, result: DistanceResult, distance: float, /) -> None: ...
    def store(self, layer: Layer, result: StoreResult, reason: str, /) -> None: ...
    def bypass(self, reason: str, /) -> None: ...
    def singleflight(self, outcome: SingleFlightOutcome, /) -> None: ...
    def cost_saved(self, layer: Layer, usd: float, /) -> None: ...
    def tokens_saved(self, layer: Layer, kind: str, tokens: int, /) -> None: ...


class NullCacheHooks:
    def lookup(self, layer: Layer, result: LookupResult, /) -> None:
        return None

    def lookup_duration(self, layer: Layer, seconds: float, /) -> None:
        return None

    def semantic_distance(self, result: DistanceResult, distance: float, /) -> None:
        return None

    def store(self, layer: Layer, result: StoreResult, reason: str, /) -> None:
        return None

    def bypass(self, reason: str, /) -> None:
        return None

    def singleflight(self, outcome: SingleFlightOutcome, /) -> None:
        return None

    def cost_saved(self, layer: Layer, usd: float, /) -> None:
        return None

    def tokens_saved(self, layer: Layer, kind: str, tokens: int, /) -> None:
        return None


# usd an upstream call with this usage would cost now, or None when the deployment has no price
type Pricer = Callable[[UsageRecord], float | None]

# what the client alias means right now (alias definition + routing policy); changing it invalidates entries
type AliasRevision = Callable[[RequestContext], str]

# extra_body.gg fields the cache acts on (C2 reports unconsumed ones as ignored)
CONSUMED_EXTENSIONS: frozenset[str] = frozenset({"cache", "cache_ttl_s", "semantic_cache", "cache_threshold"})
