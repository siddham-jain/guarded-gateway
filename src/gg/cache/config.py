from typing import Annotated, Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from gg.core.schema import StrictModel


class ExactConfig(StrictModel):
    enabled: bool = True
    min_ttl_s: PositiveInt = 60
    max_ttl_s: PositiveInt = 604_800
    max_value_bytes: PositiveInt = 262_144
    compress_over_bytes: Annotated[int, Field(ge=0)] = 1024
    singleflight: bool = True
    singleflight_wait_s: Annotated[float, Field(gt=0, le=30)] = 2.0
    singleflight_poll_s: Annotated[float, Field(gt=0, le=1)] = 0.05
    lock_ttl_ms: PositiveInt = 30_000
    post_hoc_wait_s: Annotated[float, Field(gt=0, le=60)] = 10.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_ttl_s > self.max_ttl_s:
            raise ValueError("min_ttl_s must not exceed max_ttl_s")
        if self.singleflight_poll_s > self.singleflight_wait_s:
            raise ValueError("singleflight_poll_s must not exceed singleflight_wait_s")
        return self


class EmbedderConfig(StrictModel):
    provider: Literal["fastembed", "hashing"] = "fastembed"
    name: str = "BAAI/bge-small-en-v1.5"
    dim: PositiveInt = 384


class SemanticConfig(StrictModel):
    enabled: bool = True
    embedder: EmbedderConfig = EmbedderConfig()
    # cosine distance; a request may only tighten it via gg.cache_threshold
    distance_threshold: Annotated[float, Field(ge=0, le=2)] = 0.08
    deadline_s: Annotated[float, Field(gt=0, le=5)] = 0.15
    min_user_chars: PositiveInt = 8
    max_user_chars: PositiveInt = 1200
    index: Literal["hnsw", "flat"] = "hnsw"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_user_chars > self.max_user_chars:
            raise ValueError("min_user_chars must not exceed max_user_chars")
        return self


class BreakerConfig(StrictModel):
    errors: PositiveInt = 5
    window_s: PositiveFloat = 10
    cooldown_s: PositiveFloat = 10


class BackendConfig(StrictModel):
    op_timeout_s: Annotated[float, Field(gt=0, le=5)] = 0.05
    breaker: BreakerConfig = BreakerConfig()


class CacheConfig(StrictModel):
    version: Literal[1] = 1
    exact: ExactConfig = ExactConfig()
    semantic: SemanticConfig = SemanticConfig()
    backend: BackendConfig = BackendConfig()
