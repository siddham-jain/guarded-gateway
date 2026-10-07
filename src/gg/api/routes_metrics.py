import hmac

from starlette.requests import Request
from starlette.responses import Response

from gg.api.deps import get_services
from gg.api.responses import error_body, json_response


def _unauthorized() -> Response:
    return json_response(
        error_body("A valid metrics bearer token is required.", code="invalid_api_key"),
        status=401,
        headers={"www-authenticate": 'Bearer realm="gg"', "x-should-retry": "false"},
    )


def _bearer_matches(header: str | None, expected: str) -> bool:
    if header is None:
        return False
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), expected.encode())


async def metrics(request: Request) -> Response:
    services = get_services(request)
    renderer = services.metrics_renderer
    if renderer is None or not services.settings.metrics.enabled:
        return json_response(
            error_body("Metrics are not enabled."), status=404, headers={"x-should-retry": "false"}
        )
    token = services.settings.metrics.bearer_token
    if token is not None and not _bearer_matches(
        request.headers.get("authorization"), token.get_secret_value()
    ):
        return _unauthorized()
    body, content_type = renderer()
    return Response(body, media_type=content_type, headers={"cache-control": "no-store"})
