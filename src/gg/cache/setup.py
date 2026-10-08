"""what the composition root needs: backends from config/cache.yaml, the exact stage and the semantic probe"""

from dataclasses import dataclass

import structlog
from redis.asyncio import Redis

from gg.cache.base import (
    CONSUMED_EXTENSIONS,
    AliasRevision,
    CacheBackend,
    CacheHooks,
    Embedder,
    NullCacheHooks,
    Pricer,
    SemanticIndex,
    SemanticVerifier,
)
from gg.cache.codec import Codec
from gg.cache.config import CacheConfig
from gg.cache.embedders.fastembed import FastEmbedEmbedder
from gg.cache.failopen import FailOpenCache
from gg.cache.keys import CacheKeyBuilder
from gg.cache.memory import InMemoryIndex, InMemoryResponseCache
from gg.cache.redis_exact import RedisResponseCache
from gg.cache.stages import ExactCacheStage, HitResponder, SemanticCacheProbe
from gg.cache.vector_redis import RedisVectorIndex
from gg.cache.writer import CacheWriter
from gg.config.hashing import combined_hash, section_hash
from gg.core.aio import TaskSupervisor
from gg.core.clock import Clock

log = structlog.get_logger("gg.cache")


@dataclass(frozen=True, slots=True)
class BuiltCache:
    # pipeline: [observability, guard_in_pre, guard_out, exact_stage, probes(+semantic_probe), routing]
    exact_stage: ExactCacheStage
    semantic_probe: SemanticCacheProbe | None
    hash: str
    consumed_extensions: frozenset[str]
    vector_index: RedisVectorIndex | None = None

    async def start(self) -> None:
        if self.vector_index is None:
            return
        try:
            await self.vector_index.start()
        except Exception as exc:
            # exact caching keeps working; the index stays unavailable so no request is semantic-eligible
            log.warning("cache.semantic_index_failed", error=type(exc).__name__, detail=str(exc))

    async def stop(self) -> None:
        return None


def _semantic_embedder(config: CacheConfig, embedder: Embedder | None) -> Embedder | None:
    if not config.semantic.enabled or embedder is None:
        return None
    if isinstance(embedder, FastEmbedEmbedder) and not FastEmbedEmbedder.installed():
        log.warning("cache.semantic_disabled", reason="fastembed is not installed (ml extra)")
        return None
    return embedder


def build_cache(
    config: CacheConfig,
    *,
    redis: Redis | None,
    clock: Clock,
    supervisor: TaskSupervisor,
    embedder: Embedder | None,
    metrics_hooks: CacheHooks | None = None,
    pricer: Pricer | None = None,
    alias_revision: AliasRevision | None = None,
    verifier: SemanticVerifier | None = None,
) -> BuiltCache:
    """without redis, process-local backends (dev and tests); semantic needs an embedder and FT.* or memory"""
    hooks = metrics_hooks if metrics_hooks is not None else NullCacheHooks()
    codec = Codec(
        compress_over_bytes=config.exact.compress_over_bytes, max_value_bytes=config.exact.max_value_bytes
    )
    backend: CacheBackend = (
        RedisResponseCache(redis, codec) if redis is not None else InMemoryResponseCache(codec, clock)
    )
    cache = FailOpenCache(backend, config.backend, clock)
    sem_embedder = _semantic_embedder(config, embedder)
    index: SemanticIndex | None = None
    vector_index: RedisVectorIndex | None = None
    if sem_embedder is not None:
        if redis is not None:
            vector_index = RedisVectorIndex(
                redis, embedder_name=sem_embedder.name, dim=sem_embedder.dim, algorithm=config.semantic.index
            )
            index = vector_index
        else:
            index = InMemoryIndex(clock)
    responder = HitResponder(clock, hooks, pricer)
    writer = CacheWriter(
        config,
        cache,
        clock=clock,
        supervisor=supervisor,
        hooks=hooks,
        index=index,
        embedder=sem_embedder,
        pricer=pricer,
    )
    stage = ExactCacheStage(
        config,
        cache,
        keys=CacheKeyBuilder(alias_revision),
        responder=responder,
        writer=writer,
        hooks=hooks,
        clock=clock,
        index=index,
    )
    probe = None
    if config.semantic.verifier.type != "none" and verifier is None:
        # the threshold is tuned for verified matches, so without the verifier nothing is served
        log.warning("cache.semantic_lookup_disabled", reason="the configured verifier is not available")
    elif index is not None and sem_embedder is not None:
        probe = SemanticCacheProbe(
            config,
            cache,
            index=index,
            embedder=sem_embedder,
            responder=responder,
            hooks=hooks,
            clock=clock,
            verifier=verifier,
        )
    embedder_id = f"{sem_embedder.name}:{sem_embedder.dim}" if sem_embedder is not None else "none"
    digest = combined_hash({"config": section_hash(config), "embedder": embedder_id})
    return BuiltCache(
        exact_stage=stage,
        semantic_probe=probe,
        hash=digest,
        consumed_extensions=CONSUMED_EXTENSIONS,
        vector_index=vector_index,
    )
