"""exact cache in redis: SET EX NX values, SET NX PX single-flight locks released by compare-and-delete"""

import secrets

import structlog
from redis.asyncio import Redis

from gg.cache.base import CachedResponse, PutResult
from gg.cache.codec import Codec, CorruptValueError

log = structlog.get_logger("gg.cache.redis")

_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


class RedisResponseCache:
    """raises redis errors; FailOpenCache turns them into misses"""

    def __init__(self, redis: Redis, codec: Codec) -> None:
        self._redis = redis
        self._codec = codec
        self._release = redis.register_script(_RELEASE)

    @property
    def available(self) -> bool:
        return True

    async def get(self, key: str, /) -> CachedResponse | None:
        data = await self._redis.get(key)
        if not isinstance(data, bytes):
            return None
        try:
            return self._codec.decode(data)
        except CorruptValueError:
            log.warning("cache.corrupt_value", backend="redis")
            await self._redis.unlink(key)
            return None

    async def put(
        self, key: str, value: CachedResponse, ttl_s: int, /, *, replace: bool = False
    ) -> PutResult:
        data = self._codec.encode(value)
        if not self._codec.fits(data):
            return "too_large"
        stored = await self._redis.set(key, data, ex=ttl_s, nx=not replace)
        return "stored" if stored else "exists"

    async def acquire(self, lock_key: str, ttl_ms: int, /) -> str | None:
        token = secrets.token_hex(8)
        acquired = await self._redis.set(lock_key, token, px=ttl_ms, nx=True)
        return token if acquired else None

    async def release(self, lock_key: str, token: str, /) -> None:
        await self._release(keys=[lock_key], args=[token])

    async def held(self, lock_key: str, /) -> bool:
        return bool(await self._redis.exists(lock_key))
