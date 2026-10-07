import structlog
from fastapi import FastAPI

from gg.api.deps import ApiServices
from gg.api.errors import install_exception_handlers
from gg.api.middleware import install_middleware
from gg.api.routes_chat import chat_completions
from gg.api.routes_health import healthz, readyz, version
from gg.api.routes_messages import messages
from gg.api.routes_metrics import metrics
from gg.api.routes_models import list_models, retrieve_model
from gg.api.routes_playground import playground

__all__ = ["drain", "install_exception_handlers", "install_middleware", "install_routes"]

log = structlog.get_logger("gg.api")


def install_routes(app: FastAPI) -> None:
    # never 307-redirect a POST to a trailing-slash variant
    app.router.redirect_slashes = False
    app.add_api_route("/v1/chat/completions", chat_completions, methods=["POST"])
    app.add_api_route("/v1/messages", messages, methods=["POST"])
    app.add_api_route("/v1/models", list_models, methods=["GET"])
    app.add_api_route("/v1/models/{model:path}", retrieve_model, methods=["GET"])
    app.add_api_route("/healthz", healthz, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route("/readyz", readyz, methods=["GET"], include_in_schema=False)
    app.add_api_route("/version", version, methods=["GET"], include_in_schema=False)
    app.add_api_route("/metrics", metrics, methods=["GET"], include_in_schema=False)
    app.add_api_route("/playground", playground, methods=["GET"], include_in_schema=False)


async def drain(services: ApiServices) -> None:
    """lifespan shutdown: readyz -> 503, new chat -> 503, wait for in-flight work, flush background tasks"""
    state = services.state
    state.begin_drain()
    log.info("app.draining", inflight=state.inflight)
    drained = await state.wait_idle(services.settings.server.drain_s)
    await services.supervisor.drain(services.settings.server.finalizer_timeout_s)
    log.info("app.drained", inflight=state.inflight, clean=drained)
