import time
from datetime import UTC, datetime, timedelta

import pytest

from gg.auth.keys import generate_key, hash_key
from gg.auth.resolver import CachingKeyResolver
from gg.core.clock import FakeClock
from gg.core.errors import AuthenticationError
from gg.core.keypolicy import KeyPolicy
from tests.conftest import make_key


class CountingStore:
    def __init__(self, policies: dict[str, KeyPolicy]) -> None:
        self.policies = policies
        self.calls = 0

    async def get(self, key_hash: str, /) -> KeyPolicy | None:
        self.calls += 1
        return self.policies.get(key_hash)


@pytest.fixture
def token() -> str:
    return generate_key("test").token


def _resolver(store: CountingStore, clock: FakeClock, **kwargs: float) -> CachingKeyResolver:
    return CachingKeyResolver(store, clock, allow_test_keys=True, **kwargs)  # pyright: ignore[reportArgumentType]


def _store_for(token: str, policy: KeyPolicy | None = None) -> CountingStore:
    return CountingStore({hash_key(token): policy or make_key()})


async def test_hit_after_miss(token: str, clock: FakeClock) -> None:
    store = _store_for(token)
    resolver = _resolver(store, clock)
    assert (await resolver.resolve(token)).id == "test-key"
    assert (await resolver.resolve(token)).id == "test-key"
    assert store.calls == 1
    assert resolver.lookups == {"hit": 1, "miss": 1, "negative_hit": 0}


async def test_ttl_expiry_refetches(token: str, clock: FakeClock) -> None:
    store = _store_for(token)
    resolver = _resolver(store, clock, ttl_s=60)
    await resolver.resolve(token)
    clock.advance(61)
    await resolver.resolve(token)
    assert store.calls == 2


async def test_negative_cache(clock: FakeClock) -> None:
    store = CountingStore({})
    resolver = _resolver(store, clock, negative_ttl_s=5)
    unknown = generate_key("test").token
    for _ in range(3):
        with pytest.raises(AuthenticationError) as info:
            await resolver.resolve(unknown)
        assert info.value.code == "invalid_api_key"
    assert store.calls == 1
    assert resolver.lookups["negative_hit"] == 2
    clock.advance(6)
    with pytest.raises(AuthenticationError):
        await resolver.resolve(unknown)
    assert store.calls == 2


async def test_malformed_token_never_hits_store(clock: FakeClock) -> None:
    store = CountingStore({})
    resolver = _resolver(store, clock)
    with pytest.raises(AuthenticationError):
        await resolver.resolve("gg-live-garbage")
    assert store.calls == 0


async def test_lru_eviction(clock: FakeClock) -> None:
    tokens = [generate_key("test").token for _ in range(3)]
    store = CountingStore({hash_key(t): make_key() for t in tokens})
    resolver = _resolver(store, clock, maxsize=2)
    for t in tokens:
        await resolver.resolve(t)
    assert len(resolver) == 2
    await resolver.resolve(tokens[2])
    assert store.calls == 3
    await resolver.resolve(tokens[0])
    assert store.calls == 4


async def test_disabled_key(token: str, clock: FakeClock) -> None:
    resolver = _resolver(_store_for(token, make_key(status="disabled")), clock)
    with pytest.raises(AuthenticationError) as info:
        await resolver.resolve(token)
    assert info.value.code == "api_key_disabled"


async def test_expiry_checked_on_every_request(token: str, clock: FakeClock) -> None:
    expires = clock.now() + timedelta(seconds=10)
    resolver = _resolver(_store_for(token, make_key(expires_at=expires)), clock, ttl_s=3600)
    await resolver.resolve(token)
    clock.advance(11)
    with pytest.raises(AuthenticationError) as info:
        await resolver.resolve(token)
    assert info.value.code == "api_key_expired"


async def test_test_keys_rejected_in_prod(token: str, clock: FakeClock) -> None:
    store = _store_for(token)
    resolver = CachingKeyResolver(store, clock, allow_test_keys=False)
    with pytest.raises(AuthenticationError):
        await resolver.resolve(token)
    assert store.calls == 0


async def test_cached_lookup_is_fast(token: str) -> None:
    clock = FakeClock(wall=datetime(2026, 10, 5, tzinfo=UTC).timestamp())
    resolver = _resolver(_store_for(token), clock)
    await resolver.resolve(token)
    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        await resolver.resolve(token)
    per_call_us = (time.perf_counter() - start) / n * 1e6
    # nfr-1 target is 20 us p50; a loose bound keeps this stable on slow ci runners
    assert per_call_us < 200
