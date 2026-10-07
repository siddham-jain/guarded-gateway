"""per-ip failed-auth limiter: max_failures 401s in one fixed window earn the ip a 429 until it ends"""

from typing import Protocol

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from gg.core.clock import Clock

log = structlog.get_logger("gg.auth")

NAMESPACE = "gg:authfail"
_MAX_TRACKED = 50_000


class AuthFailureLimiter(Protocol):
    async def blocked_for(self, ip: str, /) -> float | None:
        """seconds until the ip may try again, or None when it is not blocked"""
        ...

    async def record_failure(self, ip: str, /) -> None: ...


class InMemoryAuthFailureLimiter:
    """process-local fixed windows; the check is a dict lookup, so it costs nothing on the hot path"""

    def __init__(self, clock: Clock, *, max_failures: int, window_s: float) -> None:
        self._clock = clock
        self._max_failures = max_failures
        self._window_s = window_s
        # ip -> (window start on the monotonic clock, failures in that window)
        self._windows: dict[str, tuple[float, int]] = {}

    def _window(self, ip: str) -> tuple[float, int] | None:
        entry = self._windows.get(ip)
        if entry is not None and self._clock.monotonic() - entry[0] >= self._window_s:
            del self._windows[ip]
            return None
        return entry

    async def blocked_for(self, ip: str, /) -> float | None:
        entry = self._window(ip)
        if entry is None or entry[1] < self._max_failures:
            return None
        return entry[0] + self._window_s - self._clock.monotonic()

    async def record_failure(self, ip: str, /) -> None:
        self.mark(ip, None)

    def mark(self, ip: str, count: int | None) -> None:
        """count None adds one failure; an explicit count comes from a shared store"""
        entry = self._window(ip)
        start, seen = entry if entry is not None else (self._clock.monotonic(), 0)
        seen = seen + 1 if count is None else max(seen, count)
        self._windows[ip] = (start, seen)
        if seen == self._max_failures:
            log.warning("auth.ip_blocked", ip=ip, window_s=self._window_s)
        if len(self._windows) > _MAX_TRACKED:
            self._prune()

    def _prune(self) -> None:
        now = self._clock.monotonic()
        for ip in [ip for ip, (start, _) in self._windows.items() if now - start >= self._window_s]:
            del self._windows[ip]
        # still full means a flood of distinct ips; dropping the oldest half bounds memory
        if len(self._windows) > _MAX_TRACKED:
            for ip in list(self._windows)[: len(self._windows) // 2]:
                del self._windows[ip]


class RedisAuthFailureLimiter:
    """failures are counted in redis so restarts and replicas share them; checks stay local"""

    def __init__(self, redis: Redis, clock: Clock, *, max_failures: int, window_s: float) -> None:
        self._redis = redis
        self._clock = clock
        self._window_s = window_s
        self._local = InMemoryAuthFailureLimiter(clock, max_failures=max_failures, window_s=window_s)

    async def blocked_for(self, ip: str, /) -> float | None:
        return await self._local.blocked_for(ip)

    async def record_failure(self, ip: str, /) -> None:
        window = int(self._clock.time() // self._window_s)
        key = f"{NAMESPACE}:{window}:{ip}"
        try:
            count = int(await self._redis.incr(key))
            if count == 1:
                await self._redis.expire(key, max(1, round(self._window_s)))
        except (RedisError, OSError) as exc:
            # fail open to the process-local count
            log.warning("auth.failure_store_error", error=type(exc).__name__)
            self._local.mark(ip, None)
            return
        self._local.mark(ip, count)
