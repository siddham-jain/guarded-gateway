"""a cache outage never fails a request: errors and slow calls are misses, then a breaker skips the cache"""

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable

import structlog

from gg.cache.base import CacheBackend, CachedResponse, LookupResult, PutResult
from gg.cache.config import BackendConfig
from gg.core.clock import Clock

log = structlog.get_logger("gg.cache")


class Breaker:
    def __init__(self, cfg: BackendConfig, clock: Clock) -> None:
        self._cfg = cfg.breaker
        self._clock = clock
        self._errors: deque[float] = deque()
        self._open_until = 0.0

    @property
    def closed(self) -> bool:
        return self._clock.monotonic() >= self._open_until

    def failure(self) -> None:
        now = self._clock.monotonic()
        self._errors.append(now)
        while self._errors and self._errors[0] <= now - self._cfg.window_s:
            self._errors.popleft()
        if len(self._errors) >= self._cfg.errors:
            self._errors.clear()
            self._open_until = now + self._cfg.cooldown_s
            log.warning("cache.breaker_open", cooldown_s=self._cfg.cooldown_s)


class _Failed:
    def __init__(self, timeout: bool) -> None:
        self.timeout = timeout


class FailOpenCache:
    """wraps a CacheBackend; reads fail as misses, writes as no-ops, lock calls as 'nobody holds it'"""

    def __init__(self, inner: CacheBackend, cfg: BackendConfig, clock: Clock) -> None:
        self._inner = inner
        self._timeout_s = cfg.op_timeout_s
        self._breaker = Breaker(cfg, clock)

    @property
    def available(self) -> bool:
        return self._breaker.closed and self._inner.available

    async def _call[T](self, op: str, fn: Callable[[], Awaitable[T]]) -> T | _Failed:
        if not self._breaker.closed:
            return _Failed(timeout=False)
        try:
            async with asyncio.timeout(self._timeout_s):
                return await fn()
        except Exception as exc:
            self._breaker.failure()
            log.warning("cache.backend_error", op=op, error=type(exc).__name__)
            return _Failed(timeout=isinstance(exc, TimeoutError))

    async def lookup(self, key: str) -> tuple[CachedResponse | None, LookupResult]:
        out = await self._call("get", lambda: self._inner.get(key))
        if isinstance(out, _Failed):
            return None, "timeout" if out.timeout else "error"
        return out, "miss" if out is None else "hit"

    async def get(self, key: str, /) -> CachedResponse | None:
        value, _ = await self.lookup(key)
        return value

    async def put(
        self, key: str, value: CachedResponse, ttl_s: int, /, *, replace: bool = False
    ) -> PutResult:
        out = await self._call("put", lambda: self._inner.put(key, value, ttl_s, replace=replace))
        return "error" if isinstance(out, _Failed) else out

    async def acquire(self, lock_key: str, ttl_ms: int, /) -> str | None:
        out = await self._call("acquire", lambda: self._inner.acquire(lock_key, ttl_ms))
        return None if isinstance(out, _Failed) else out

    async def release(self, lock_key: str, token: str, /) -> None:
        await self._call("release", lambda: self._inner.release(lock_key, token))

    async def held(self, lock_key: str, /) -> bool:
        out = await self._call("held", lambda: self._inner.held(lock_key))
        return False if isinstance(out, _Failed) else out
