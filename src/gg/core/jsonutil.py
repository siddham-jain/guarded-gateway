import hashlib
from decimal import Decimal
from typing import Any

import orjson

type JSONValue = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None


def _default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not json serialisable: {type(obj).__name__}")


def dumps(obj: Any) -> bytes:
    return orjson.dumps(obj, default=_default)


def dumps_str(obj: Any) -> str:
    return dumps(obj).decode()


def loads(data: bytes | bytearray | memoryview | str) -> Any:
    return orjson.loads(data)


def canonical_json(obj: Any) -> bytes:
    return orjson.dumps(obj, default=_default, option=orjson.OPT_SORT_KEYS)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()
