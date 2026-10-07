from collections.abc import Iterable
from dataclasses import dataclass

from gg.core.errors import InvalidRequestError, NotFoundError, PermissionDeniedError
from gg.core.keypolicy import KeyPolicy
from gg.core.normalize import Normalization
from gg.core.schema import ChatRequest, GGExtensions
from gg.providers.base import DeploymentRef, ModelCatalog

ROUTER_ALIAS = "gg/auto"
_IGNORED_NORMALIZATIONS = frozenset({"stream_options_dropped"})


@dataclass(frozen=True, slots=True)
class AuthorizedRequest:
    request: ChatRequest
    ignored_params: frozenset[str]


def _not_allowed(model: str) -> PermissionDeniedError:
    return PermissionDeniedError(
        f"The model `{model}` does not exist or you do not have access to it.",
        param="model",
        code="model_not_allowed",
    )


def _check_model(req: ChatRequest, key: KeyPolicy, catalog: ModelCatalog) -> None:
    if not key.allows_model(req.model):
        raise _not_allowed(req.model)
    resolution = catalog.resolve(req.model)
    if resolution is None:
        raise NotFoundError(
            f"The model `{req.model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found",
        )
    # alias members are filtered by the same flags at routing time
    if isinstance(resolution, DeploymentRef) and not key.allows_deployment(resolution.deployment):
        raise _not_allowed(req.model)


def _cap(value: int, limit: int | None, param: str) -> None:
    if limit is not None and value > limit:
        raise InvalidRequestError(
            f"Invalid value for '{param}': {value} exceeds this key's limit of {limit}.", param=param
        )


def _check_caps(req: ChatRequest, key: KeyPolicy) -> None:
    limits = key.limits
    _cap(len(req.messages), limits.max_messages, "messages")
    _cap(len(req.tools or ()), limits.max_tools, "tools")
    _cap(req.n, limits.max_n, "n")
    if req.max_completion_tokens is not None:
        _cap(req.max_completion_tokens, limits.max_completion_tokens, "max_completion_tokens")


def _strip_params(req: ChatRequest, key: KeyPolicy) -> tuple[ChatRequest, set[str]]:
    extra = req.model_extra or {}
    stripped: set[str] = set()
    if "service_tier" in extra and not key.flags.allow_service_tier:
        stripped.add("service_tier")
    if extra.get("store") is True and not key.flags.allow_store:
        stripped.add("store")
    if not stripped:
        return req, stripped
    payload = req.model_dump(by_alias=True, exclude_unset=True, exclude=stripped)
    return ChatRequest.model_validate(payload), stripped


def _check_extensions(
    req: ChatRequest, ext: GGExtensions, key: KeyPolicy, consumed: frozenset[str]
) -> set[str]:
    if ext.route_threshold is not None and req.model == ROUTER_ALIAS:
        if not key.routing.allow_request_threshold:
            raise PermissionDeniedError(
                "This key may not override the routing threshold.",
                param="gg.route_threshold",
                code="gg_override_not_allowed",
            )
        low, high = key.routing.threshold_bounds
        if not low <= ext.route_threshold <= high:
            raise InvalidRequestError(
                f"gg.route_threshold must lie within [{low}, {high}] for this key.",
                param="gg.route_threshold",
            )
    if ext.guardrails is not None and not key.guardrails.allow_request_tightening:
        raise PermissionDeniedError(
            "This key may not change guardrail behaviour per request.",
            param="gg.guardrails",
            code="gg_override_not_allowed",
        )
    if ext.cache_ttl_s is not None and ext.cache_ttl_s > key.cache.max_ttl_s:
        raise InvalidRequestError(
            f"gg.cache_ttl_s may not exceed {key.cache.max_ttl_s} for this key.", param="gg.cache_ttl_s"
        )
    ignored = {f"gg.{name}" for name in ext.model_fields_set if name not in consumed}
    if ext.route_threshold is not None and req.model != ROUTER_ALIAS:
        ignored.add("gg.route_threshold")
    return ignored


def _ignored_by_normalization(applied: Iterable[Normalization]) -> set[str]:
    return {n.param for n in applied if n.rule in _IGNORED_NORMALIZATIONS}


def authorize_request(
    req: ChatRequest,
    key: KeyPolicy,
    catalog: ModelCatalog,
    *,
    normalizations: Iterable[Normalization] = (),
    consumed_extensions: frozenset[str] = frozenset(),
) -> AuthorizedRequest:
    """per-key checks before the pipeline: allowlist, existence, flags, caps, stripped params, extensions"""
    _check_model(req, key, catalog)
    _check_caps(req, key)
    working, ignored = _strip_params(req, key)
    ignored |= _ignored_by_normalization(normalizations)
    if req.gg is not None:
        ignored |= _check_extensions(req, req.gg, key, consumed_extensions)
    return AuthorizedRequest(request=working, ignored_params=frozenset(ignored))
