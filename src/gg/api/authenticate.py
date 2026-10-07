import structlog
from starlette.requests import Request

from gg.api.deps import ApiServices
from gg.auth.keys import key_prefix, parse_bearer
from gg.core.context import StageTimings
from gg.core.errors import AuthenticationError
from gg.core.keypolicy import KeyPolicy

log = structlog.get_logger("gg.api")


def _token(request: Request, *, allow_api_key_header: bool) -> str:
    authorization = request.headers.getlist("authorization")
    api_keys = request.headers.getlist("x-api-key") if allow_api_key_header else []
    if authorization or not api_keys:
        return parse_bearer(authorization)
    if len(api_keys) > 1 or not api_keys[0].strip():
        raise AuthenticationError("Send exactly one non-empty x-api-key header.", code="invalid_api_key")
    return api_keys[0].strip()


async def authenticate(
    request: Request, services: ApiServices, *, allow_api_key_header: bool = False
) -> KeyPolicy:
    """bearer (or x-api-key on anthropic-style routes) -> KeyPolicy, before the body is read"""
    timings: StageTimings = request.state.timings
    token: str | None = None
    try:
        with timings.measure("auth"):
            token = _token(request, allow_api_key_header=allow_api_key_header)
            key = await services.keys.resolve(token)
    except AuthenticationError as exc:
        prefix = key_prefix(token) if token else None
        log.info("auth.failed", reason=exc.code, prefix=prefix)
        raise
    timings.mark("authenticated")
    structlog.contextvars.bind_contextvars(key_id=key.id)
    request.state.key = key
    return key
