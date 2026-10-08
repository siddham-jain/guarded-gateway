import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import fakeredis
import pytest
from redis.asyncio import Redis

from gg.cache.base import CacheBackend, CachedResponse, SemanticTags
from gg.cache.codec import RAW, ZLIB, Codec, CorruptValueError
from gg.cache.memory import InMemoryIndex, InMemoryResponseCache, cosine_distance
from gg.cache.redis_exact import RedisResponseCache
from gg.cache.vector_redis import RedisVectorIndex, parse_search
from gg.core.clock import FakeClock
from tests.unit.cache.conftest import REDIS_URL_ENV
from tests.unit.cache.support import USAGE, codec


def entry(content: str = "Paris.", **fields: Any) -> CachedResponse:
    return CachedResponse(
        content=content,
        finish_reason="stop",
        usage=USAGE,
        response_model="echo-1",
        created_at=1.0,
        request_id="req_1",
        **fields,
    )


def test_codec_round_trip_and_compression() -> None:
    c = codec()
    small, big = entry(), entry("x" * 5000)
    small_data, big_data = c.encode(small), c.encode(big)
    assert small_data[:1] == RAW
    assert big_data[:1] == ZLIB
    assert len(big_data) < 1000
    assert c.decode(small_data) == small
    assert c.decode(big_data) == big


@pytest.mark.parametrize("data", [b"\x07junk", b"\x00{not json", b"\x01not zlib", b'\x00{"v": 1}'])
def test_codec_rejects_corrupt_values(data: bytes) -> None:
    with pytest.raises(CorruptValueError):
        Codec.decode(data)


def test_codec_size_limit() -> None:
    c = Codec(compress_over_bytes=10_000, max_value_bytes=100)
    assert not c.fits(c.encode(entry("y" * 500)))


@pytest.fixture
async def fake_redis() -> AsyncIterator[Redis]:
    client = fakeredis.FakeAsyncRedis()
    yield client
    await client.aclose()


type Factory = Callable[[FakeClock, Redis], CacheBackend]

BACKENDS: dict[str, Factory] = {
    "memory": lambda clock, _: InMemoryResponseCache(codec(), clock),
    "redis": lambda _, redis: RedisResponseCache(redis, codec()),
}


@pytest.fixture(params=sorted(BACKENDS))
def backend(request: pytest.FixtureRequest, fake_redis: Redis) -> CacheBackend:
    return BACKENDS[request.param](FakeClock(), fake_redis)


async def test_contract_get_put(backend: CacheBackend) -> None:
    assert await backend.get("k") is None
    assert await backend.put("k", entry(), 60) == "stored"
    assert await backend.get("k") == entry()
    assert await backend.put("k", entry("other"), 60) == "exists"
    assert await backend.get("k") == entry()
    assert await backend.put("k", entry("other"), 60, replace=True) == "stored"
    got = await backend.get("k")
    assert got is not None
    assert got.content == "other"


async def test_contract_too_large(backend: CacheBackend) -> None:
    huge = entry("".join(chr(0x4E00 + (i * 7919) % 20000) for i in range(200_000)))
    assert await backend.put("k", huge, 60) == "too_large"
    assert await backend.get("k") is None


async def test_contract_lock(backend: CacheBackend) -> None:
    token = await backend.acquire("k:lk", 30_000)
    assert token is not None
    assert await backend.held("k:lk")
    assert await backend.acquire("k:lk", 30_000) is None
    await backend.release("k:lk", "wrong-token")
    assert await backend.held("k:lk")
    await backend.release("k:lk", token)
    assert not await backend.held("k:lk")


async def test_memory_ttl_expiry() -> None:
    clock = FakeClock()
    cache = InMemoryResponseCache(codec(), clock)
    await cache.put("k", entry(), 60)
    lock = await cache.acquire("k:lk", 1000)
    assert lock is not None
    clock.advance(61)
    assert await cache.get("k") is None
    assert not await cache.held("k:lk")
    assert await cache.put("k", entry(), 60) == "stored"


async def test_redis_sets_ttl_and_unlinks_corrupt_values(fake_redis: Redis) -> None:
    cache = RedisResponseCache(fake_redis, codec())
    await cache.put("k", entry(), 120)
    ttl = await fake_redis.ttl("k")
    assert 0 < ttl <= 120
    await fake_redis.set("bad", b"\x00garbage")
    assert await cache.get("bad") is None
    assert not await fake_redis.exists("bad")


def tags(scope: str = "s", num_sig: str = "n") -> SemanticTags:
    return SemanticTags(scope, "a", "r", "p", "y", "q", num_sig)


async def test_memory_index_nearest_with_tag_isolation_and_ttl() -> None:
    clock = FakeClock()
    index = InMemoryIndex(clock)
    await index.add([1.0, 0.0], tags(), "near", 60, "near prompt")
    await index.add([0.0, 1.0], tags(), "far", 60, "far prompt")
    await index.add([1.0, 0.0], tags(scope="other"), "foreign", 60, "foreign prompt")
    match = await index.search([0.9, 0.1], tags())
    assert match is not None
    assert (match.exact_key, match.text) == ("near", "near prompt")
    assert match.distance == pytest.approx(cosine_distance([0.9, 0.1], [1.0, 0.0]))
    assert await index.search([1.0, 0.0], tags(scope="nobody")) is None
    assert await index.search([1.0, 0.0], tags(num_sig="different")) is None
    clock.advance(61)
    assert await index.search([1.0, 0.0], tags()) is None
    assert len(index) == 0


def test_cosine_distance_basics() -> None:
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_vector_query_filters_on_every_tag() -> None:
    query = RedisVectorIndex.query(tags())
    assert query == (
        "(@scope:{s} @alias:{a} @alias_rev:{r} @policy_sha:{p} @system_sha:{y} @params_sha:{q} @num_sig:{n})"
        "=>[KNN 1 @vec $B AS dist]"
    )


def test_index_namespace_embeds_embedder_and_dim(fake_redis: Redis) -> None:
    index = RedisVectorIndex(fake_redis, embedder_name="bge-small-en-v1.5", dim=384)
    assert index.index_name == "gg:sem:v1:bge-small-en-v1.5:384"
    assert index.prefix == "gg:sv:v1:bge-small-en-v1.5:384:"


@pytest.mark.parametrize(
    "raw",
    [
        [1, b"gg:sv:x", [b"dist", b"0.0312", b"exact_key", b"gg:c:v1:g:abc"]],
        {
            b"total_results": 1,
            b"results": [
                {b"id": b"gg:sv:x", b"extra_attributes": {b"dist": b"0.0312", b"exact_key": b"gg:c:v1:g:abc"}}
            ],
        },
        {
            "total_results": 1,
            "results": [{"extra_attributes": {"dist": "0.0312", "exact_key": "gg:c:v1:g:abc"}}],
        },
    ],
)
def test_parse_search_resp2_and_resp3(raw: Any) -> None:
    match = parse_search(raw)
    assert match is not None
    assert match.exact_key == "gg:c:v1:g:abc"
    assert match.distance == pytest.approx(0.0312)


@pytest.mark.parametrize("raw", [[0], [], {"total_results": 0, "results": []}, [1, b"k", [b"dist", b"0.1"]]])
def test_parse_search_empty(raw: Any) -> None:
    assert parse_search(raw) is None


async def test_index_without_query_engine_stays_unavailable(fake_redis: Redis) -> None:
    index = RedisVectorIndex(fake_redis, embedder_name="hashing", dim=8)
    await index.start()
    assert not index.available


@pytest.mark.redis
async def test_real_redis_vector_index_round_trip() -> None:
    redis = Redis.from_url(os.environ[REDIS_URL_ENV])
    index = RedisVectorIndex(redis, embedder_name="test-embedder", dim=2, algorithm="flat")
    try:
        await index.start()
        assert index.available
        await index.add([1.0, 0.0], tags(), "gg:c:v1:test:near", 60)
        await index.add([0.0, 1.0], tags(), "gg:c:v1:test:far", 60)
        await index.add([1.0, 0.0], tags(scope="other"), "gg:c:v1:test:foreign", 60)
        match = await index.search([0.9, 0.1], tags())
        assert match is not None
        assert match.exact_key == "gg:c:v1:test:near"
        assert match.distance == pytest.approx(cosine_distance([0.9, 0.1], [1.0, 0.0]), abs=1e-5)
        assert await index.search([1.0, 0.0], tags(scope="nobody")) is None
    finally:
        await redis.execute_command("FT.DROPINDEX", index.index_name, "DD")
        await redis.aclose()
