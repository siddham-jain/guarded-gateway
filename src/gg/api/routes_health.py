import asyncio
from typing import Literal

import structlog
from starlette.requests import Request
from starlette.responses import Response

from gg.api.deps import ServerState, get_services
from gg.api.responses import json_response
from gg.core.lifecycle import HealthCheck, HealthStatus

log = structlog.get_logger("gg.api")

CHECK_TIMEOUT_S = 0.25

type Readiness = Literal["ready", "not_ready", "draining"]


async def run_check(check: HealthCheck) -> HealthStatus:
    name = str(getattr(check, "name", type(check).__name__))
    try:
        async with asyncio.timeout(CHECK_TIMEOUT_S):
            return await check.check()
    except TimeoutError:
        return HealthStatus(name=name, status="down", detail="timed out")
    except Exception as exc:
        log.warning("health.check_failed", check=name, error=repr(exc))
        return HealthStatus(name=name, status="down", detail="check raised")


def readiness(state: ServerState, results: list[HealthStatus]) -> Readiness:
    if state.draining:
        return "draining"
    if not state.ready or any(r.gating and r.status == "down" for r in results):
        return "not_ready"
    return "ready"


async def healthz(request: Request) -> Response:
    return json_response({"status": "ok"})


async def readyz(request: Request) -> Response:
    services = get_services(request)
    results = list(await asyncio.gather(*(run_check(c) for c in services.health_checks)))
    status = readiness(services.state, results)
    body = {
        "status": status,
        "checks": {r.name: {"status": r.status, "detail": r.detail, "gating": r.gating} for r in results},
        "config_hash": services.config_hash,
    }
    return json_response(body, status=200 if status == "ready" else 503)


async def version(request: Request) -> Response:
    services = get_services(request)
    info = services.build_info
    return json_response(
        {
            "version": info.version,
            "git_sha": info.git_sha,
            "image_digest": info.image_digest,
            "built_at": info.built_at,
            "config_hash": services.config_hash,
        }
    )
