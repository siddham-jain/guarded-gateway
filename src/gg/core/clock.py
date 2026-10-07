import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def time(self) -> float: ...
    def now(self) -> datetime: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def time(self) -> float:
        return time.time()

    def now(self) -> datetime:
        return datetime.now(UTC)


class FakeClock:
    def __init__(self, start: float = 1_000.0, wall: float = 1_790_000_000.0) -> None:
        self._mono = start
        self._wall = wall

    def monotonic(self) -> float:
        return self._mono

    def time(self) -> float:
        return self._wall

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._wall, UTC)

    def advance(self, seconds: float) -> None:
        self._mono += seconds
        self._wall += seconds
