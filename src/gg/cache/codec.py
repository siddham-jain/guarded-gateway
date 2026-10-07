"""value encoding: one codec byte, then orjson, zlib-compressed above a size threshold"""

import zlib

import orjson
from pydantic import ValidationError

from gg.cache.base import CachedResponse

RAW = b"\x00"
ZLIB = b"\x01"


class CorruptValueError(ValueError):
    pass


class Codec:
    def __init__(self, *, compress_over_bytes: int, max_value_bytes: int) -> None:
        self._compress_over = compress_over_bytes
        self.max_value_bytes = max_value_bytes

    def encode(self, value: CachedResponse) -> bytes:
        raw = orjson.dumps(value.model_dump(mode="json", exclude_none=True))
        if len(raw) > self._compress_over:
            return ZLIB + zlib.compress(raw, 6)
        return RAW + raw

    def fits(self, data: bytes) -> bool:
        return len(data) <= self.max_value_bytes

    @staticmethod
    def decode(data: bytes) -> CachedResponse:
        header, body = data[:1], data[1:]
        try:
            if header == ZLIB:
                body = zlib.decompress(body)
            elif header != RAW:
                raise CorruptValueError(f"unknown codec byte {header!r}")
            return CachedResponse.model_validate(orjson.loads(body))
        except (zlib.error, orjson.JSONDecodeError, ValidationError) as exc:
            raise CorruptValueError(str(exc)) from exc
