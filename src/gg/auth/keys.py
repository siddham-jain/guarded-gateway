import hashlib
import re
import secrets
import string
import zlib
from dataclasses import dataclass
from typing import Literal

from gg.core.errors import AuthenticationError

type KeyEnv = Literal["live", "test"]

_ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase
_BODY_LEN = 43
_CHECK_LEN = 6
_MAX_TOKEN_LEN = 256

KEY_RE = re.compile(r"^gg-(live|test)-([0-9A-Za-z]{43})([0-9A-Za-z]{6})$")
PREFIX_LEN = 12


@dataclass(frozen=True, slots=True)
class GeneratedKey:
    token: str
    hash: str
    prefix: str


def _base62(value: int, width: int) -> str:
    digits: list[str] = []
    while value:
        value, rem = divmod(value, 62)
        digits.append(_ALPHABET[rem])
    return "".join(reversed(digits)).rjust(width, "0")


def _checksum(body: str) -> str:
    return _base62(zlib.crc32(body.encode("ascii")), _CHECK_LEN)


def generate_key(env: KeyEnv = "live") -> GeneratedKey:
    body = _base62(int.from_bytes(secrets.token_bytes(32)), _BODY_LEN)
    token = f"gg-{env}-{body}{_checksum(body)}"
    return GeneratedKey(token=token, hash=hash_key(token), prefix=key_prefix(token))


def hash_key(token: str) -> str:
    # a fast hash is right here: the secret is 256-bit random, so a slow kdf adds latency, not security
    return "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()


def key_prefix(token: str) -> str:
    return token[:PREFIX_LEN]


def is_well_formed(token: str) -> bool:
    match = KEY_RE.match(token)
    return match is not None and _checksum(match.group(2)) == match.group(3)


def check_format(token: str, *, allow_test_keys: bool) -> None:
    if not is_well_formed(token):
        raise AuthenticationError("Incorrect API key provided.", code="invalid_api_key")
    if not allow_test_keys and token.startswith("gg-test-"):
        raise AuthenticationError("Test keys are not accepted here.", code="invalid_api_key")


def parse_bearer(values: list[str]) -> str:
    """exactly one authorization header, scheme case-insensitive, one token; the token is never logged"""
    if not values:
        raise AuthenticationError(
            "You didn't provide an API key. Send it as 'Authorization: Bearer gg-...'.",
            code="missing_api_key",
        )
    if len(values) > 1:
        raise AuthenticationError("Send exactly one Authorization header.", code="invalid_api_key")
    parts = values[0].split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise AuthenticationError(
            "Malformed Authorization header; expected 'Bearer <key>'.", code="invalid_api_key"
        )
    token = parts[1]
    if len(token) > _MAX_TOKEN_LEN:
        raise AuthenticationError("Incorrect API key provided.", code="invalid_api_key")
    if not token.startswith("gg-"):
        raise AuthenticationError(
            "Incorrect API key provided. GG keys start with gg-.", code="invalid_api_key"
        )
    return token
