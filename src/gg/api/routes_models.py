from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from gg.api.authenticate import authenticate
from gg.api.deps import get_services
from gg.api.responses import json_response
from gg.core.errors import NotFoundError
from gg.core.keypolicy import KeyPolicy
from gg.providers.base import DeploymentRef, ModelCatalog, PublicModel

_CACHE_CONTROL = {"cache-control": "private, max-age=60"}


def visible_to(key: KeyPolicy, catalog: ModelCatalog) -> Callable[[str], bool]:
    def allowed(model_id: str) -> bool:
        if not key.allows_model(model_id):
            return False
        resolution = catalog.resolve(model_id)
        return not isinstance(resolution, DeploymentRef) or key.allows_deployment(resolution.deployment)

    return allowed


def model_entry(model: PublicModel) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model.id,
        "object": "model",
        "created": model.created,
        "owned_by": model.owned_by,
        "shutdown_date": model.shutdown_date,
    }
    if model.context_window is not None:
        entry["gg"] = {"context_window": model.context_window}
    return entry


async def list_models(request: Request) -> Response:
    services = get_services(request)
    key = await authenticate(request, services)
    models = services.catalog.list_public(visible_to(key, services.catalog))
    body = {"object": "list", "data": [model_entry(m) for m in models]}
    return json_response(body, headers=_CACHE_CONTROL)


async def retrieve_model(request: Request, model: str) -> Response:
    services = get_services(request)
    key = await authenticate(request, services)
    models = services.catalog.list_public(visible_to(key, services.catalog))
    found = next((m for m in models if m.id == model), None)
    if found is None:
        raise NotFoundError(
            f"The model `{model}` does not exist or you do not have access to it.", param="model"
        )
    return json_response(model_entry(found), headers=_CACHE_CONTROL)
