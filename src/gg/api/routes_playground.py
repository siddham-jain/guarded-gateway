from functools import cache
from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from gg.api.deps import get_services

_PAGE = Path(__file__).with_name("playground.html")
_HEADERS = {
    "cache-control": "no-store",
    "content-security-policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


@cache
def _page() -> bytes:
    return _PAGE.read_bytes()


async def playground(request: Request) -> Response:
    # a demo surface, not an api: hidden in prod
    if get_services(request).settings.env == "prod":
        raise HTTPException(status_code=404)
    return Response(_page(), media_type="text/html; charset=utf-8", headers=_HEADERS)
