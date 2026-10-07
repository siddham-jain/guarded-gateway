import secrets
import string

_ALPHABET = string.ascii_letters + string.digits


def _random(n: int) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(n))


def new_request_id() -> str:
    return "req_" + _random(24)


def completion_id(request_id: str) -> str:
    # stable across retries, fallbacks and cache replays of the same request
    return "chatcmpl-" + request_id.removeprefix("req_")


def new_trace_id() -> str:
    # w3c trace id: 16 random bytes as lowercase hex
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)
