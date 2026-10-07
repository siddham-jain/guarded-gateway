from collections import OrderedDict

from gg.core.clock import Clock


class TTLCache[V]:
    """bounded in-process lru with a per-entry ttl; per replica by design"""

    def __init__(self, max_entries: int, ttl_s: float, clock: Clock) -> None:
        self._max = max_entries
        self._ttl = ttl_s
        self._clock = clock
        self._items: OrderedDict[str, tuple[float, V]] = OrderedDict()

    def get(self, key: str) -> V | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if expires <= self._clock.monotonic():
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: str, value: V) -> None:
        self._items[key] = (self._clock.monotonic() + self._ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
