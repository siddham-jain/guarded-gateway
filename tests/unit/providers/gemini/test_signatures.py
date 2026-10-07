from gg.core.clock import FakeClock
from gg.providers.state.memory import InMemoryStateStore
from gg.providers.state.signatures import NAMESPACE, SignatureStore


async def test_round_trip_is_key_scoped() -> None:
    store = SignatureStore(InMemoryStateStore(clock=FakeClock()))
    await store.save("key-a", {"fc_1": "sig-1", "fc_2": "sig-2"})
    assert await store.lookup("key-a", ["fc_1", "fc_2", "fc_3"]) == {"fc_1": "sig-1", "fc_2": "sig-2"}
    assert await store.lookup("key-b", ["fc_1"]) == {}
    assert await store.lookup("key-a", []) == {}


async def test_entries_expire_after_the_ttl() -> None:
    clock = FakeClock()
    store = SignatureStore(InMemoryStateStore(clock=clock), ttl_s=60)
    await store.save("k", {"fc": "sig"})
    clock.advance(59)
    assert await store.lookup("k", ["fc"]) == {"fc": "sig"}
    clock.advance(2)
    assert await store.lookup("k", ["fc"]) == {}


async def test_raw_keys_use_the_signature_namespace() -> None:
    backing = InMemoryStateStore(clock=FakeClock())
    await SignatureStore(backing).save("k", {"fc": "sig"})
    assert await backing.get_many(NAMESPACE, ["k:fc"]) == {"k:fc": b"sig"}
    await SignatureStore(backing).save("k", {})
