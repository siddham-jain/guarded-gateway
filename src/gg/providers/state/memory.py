from collections import OrderedDict
from collections.abc import Mapping, Sequence

from gg.core.clock import Clock, SystemClock


class InMemoryStateStore:
    """lru + ttl store for tests and single-process dev without redis"""

    def __init__(self, *, max_items: int = 10_000, clock: Clock | None = None) -> None:
        self._items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._max = max_items
        self._clock = clock or SystemClock()

    async def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, bytes]:
        now = self._clock.monotonic()
        found: dict[str, bytes] = {}
        for key in keys:
            full = f"{namespace}:{key}"
            entry = self._items.get(full)
            if entry is None:
                continue
            expires, value = entry
            if expires <= now:
                del self._items[full]
                continue
            self._items.move_to_end(full)
            found[key] = value
        return found

    async def put_many(self, namespace: str, items: Mapping[str, bytes], ttl_s: int) -> None:
        expires = self._clock.monotonic() + ttl_s
        for key, value in items.items():
            full = f"{namespace}:{key}"
            self._items[full] = (expires, value)
            self._items.move_to_end(full)
        while len(self._items) > self._max:
            self._items.popitem(last=False)
