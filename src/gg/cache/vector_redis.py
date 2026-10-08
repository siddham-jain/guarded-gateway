"""semantic index on the redis 8 query engine via FT.* directly (no redisvl); knn 1, caller thresholds"""

import struct
from collections.abc import Sequence
from typing import Any

import structlog
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from redis.typing import EncodableT, FieldT

from gg.cache.base import SemanticMatch, SemanticTags
from gg.core.jsonutil import sha256_hex

log = structlog.get_logger("gg.cache.vector")

_TAG_FIELDS = ("scope", "alias", "alias_rev", "policy_sha", "system_sha", "params_sha", "num_sig")


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in name.lower())


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class RedisVectorIndex:
    """index and key prefix embed the embedder name and dim, so vectors of two models never mix"""

    def __init__(
        self,
        redis: Redis,
        *,
        embedder_name: str,
        dim: int,
        algorithm: str = "hnsw",
    ) -> None:
        self._redis = redis
        self._dim = dim
        self._algorithm = algorithm.upper()
        namespace = f"v1:{_slug(embedder_name)}:{dim}"
        self.index_name = f"gg:sem:{namespace}"
        self.prefix = f"gg:sv:{namespace}:"
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def start(self) -> None:
        """creates the index when missing; a redis without the query engine disables the semantic layer"""
        try:
            await self._redis.execute_command("FT.INFO", self.index_name)
        except ResponseError as exc:
            message = str(exc).lower()
            if "unknown command" in message:
                log.warning("cache.semantic_unavailable", reason="redis has no FT.* commands")
                return
            await self._create()
        self._available = True
        log.info("cache.semantic_index_ready", index=self.index_name, algorithm=self._algorithm)

    async def _create(self) -> None:
        schema: list[str] = []
        for name in _TAG_FIELDS:
            schema += [name, "TAG"]
        schema += ["created_at", "NUMERIC"]
        schema += ["vec", "VECTOR", self._algorithm, "6", "TYPE", "FLOAT32", "DIM", str(self._dim)]
        schema += ["DISTANCE_METRIC", "COSINE"]
        await self._redis.execute_command(
            "FT.CREATE", self.index_name, "ON", "HASH", "PREFIX", "1", self.prefix, "SCHEMA", *schema
        )

    @staticmethod
    def query(tags: SemanticTags) -> str:
        # tag values are hex digests, so they need no escaping
        filters = " ".join(f"@{name}:{{{value}}}" for name, value in tags.as_dict().items())
        return f"({filters})=>[KNN 1 @vec $B AS dist]"

    async def search(self, vector: Sequence[float], tags: SemanticTags, /) -> SemanticMatch | None:
        raw: Any = await self._redis.execute_command(
            "FT.SEARCH",
            self.index_name,
            self.query(tags),
            "PARAMS",
            "2",
            "B",
            pack(vector),
            "RETURN",
            "3",
            "dist",
            "exact_key",
            "text",
            "SORTBY",
            "dist",
            "LIMIT",
            "0",
            "1",
            "DIALECT",
            "2",
        )
        return parse_search(raw)

    async def add(
        self, vector: Sequence[float], tags: SemanticTags, exact_key: str, ttl_s: int, text: str, /
    ) -> None:
        key = self.prefix + sha256_hex(exact_key)[:32]
        fields: dict[FieldT, EncodableT] = {
            **tags.as_dict(),
            "exact_key": exact_key,
            "text": text,
            "vec": pack(vector),
        }
        async with self._redis.pipeline(transaction=False) as pipe:
            pipe.hset(key, mapping=fields)
            pipe.expire(key, ttl_s)
            await pipe.execute()


def _get(mapping: Any, name: str) -> Any:
    return mapping.get(name, mapping.get(name.encode()))


def parse_search(raw: Any) -> SemanticMatch | None:
    """resp3 map {results: [{extra_attributes: {...}}]} or resp2 [total, key, [field, value, ...]]"""
    if isinstance(raw, dict):
        results = _get(raw, "results") or []
        if not results:
            return None
        attrs = _get(results[0], "extra_attributes") or {}
        fields = {_text(k): _text(v) for k, v in attrs.items()}
    else:
        if not raw or int(raw[0]) == 0 or len(raw) < 3:
            return None
        flat = list(raw[2])
        fields = {_text(flat[i]): _text(flat[i + 1]) for i in range(0, len(flat) - 1, 2)}
    if "exact_key" not in fields or "dist" not in fields:
        return None
    return SemanticMatch(
        exact_key=fields["exact_key"], distance=float(fields["dist"]), text=fields.get("text")
    )
