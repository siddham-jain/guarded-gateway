import re

import structlog

from gg.core.context import RequestContext

log = structlog.get_logger("gg.api")

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("x-content-type-options", "nosniff"),
    ("referrer-policy", "no-referrer"),
    ("x-frame-options", "DENY"),
)
SSE_HEADERS: dict[str, str] = {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    "x-accel-buffering": "no",
}
EXPOSED_HEADERS: tuple[str, ...] = (
    "x-request-id",
    "x-gg-trace-id",
    "x-should-retry",
    "retry-after",
    "server-timing",
    "x-gg-provider",
    "x-gg-model",
    "x-gg-upstream-request-id",
    "x-gg-attempts",
    "x-gg-fallback",
    "x-gg-route",
    "x-gg-route-reason",
    "x-gg-cache",
    "x-gg-config",
    "x-gg-ignored-params",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

_STAGE_HEADER_RE = re.compile(r"^(x-gg-|x-ratelimit-|retry-after$)")
_MAX_VALUE_LEN = 512


def _valid_value(value: str) -> bool:
    return value.isascii() and value.isprintable() and len(value) <= _MAX_VALUE_LEN


def stage_headers(ctx: RequestContext) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name, value in ctx.response_headers.items():
        name = raw_name.lower()
        if not _STAGE_HEADER_RE.match(name) or not _valid_value(value):
            log.warning("response_header.dropped", header=name)
            continue
        out[name] = value
    return out


def server_timing(ctx: RequestContext) -> str:
    ctx.timings.mark("response_start")
    timing = ctx.timings.server_timing()
    total = ctx.timings.between("received", "response_start")
    if total is None:
        return timing
    gateway = max(0.0, total - ctx.timings.durations.get("terminal", 0.0))
    return f"{timing}, gw;dur={gateway * 1000:.2f}" if timing else f"gw;dur={gateway * 1000:.2f}"


def _served_headers(ctx: RequestContext) -> dict[str, str]:
    out: dict[str, str] = {}
    served = ctx.served_by
    if served is not None:
        out["x-gg-provider"] = served.provider
        out["x-gg-model"] = served.id
        if ctx.route is not None and ctx.route.plan.entries:
            first = ctx.route.plan.entries[0].deployment
            out["x-gg-fallback"] = "true" if first.id != served.id else "false"
    if ctx.attempts:
        out["x-gg-attempts"] = str(len(ctx.attempts))
        upstream_id = next(
            (a.upstream_request_id for a in reversed(ctx.attempts) if a.upstream_request_id), None
        )
        if upstream_id and _valid_value(upstream_id):
            out["x-gg-upstream-request-id"] = upstream_id
    return out


def build_response_headers(
    ctx: RequestContext, *, streaming: bool, early_commit: bool = False
) -> dict[str, str]:
    """x-gg-* headers known at commit time; an early sse commit predates the served deployment"""
    headers: dict[str, str] = {}
    if ctx.config_hash:
        headers["x-gg-config"] = ctx.config_hash[:12]
    if not early_commit:
        headers.update(_served_headers(ctx))
    if ctx.ignored_params:
        headers["x-gg-ignored-params"] = ",".join(sorted(ctx.ignored_params))
    headers.update(stage_headers(ctx))
    if timing := server_timing(ctx):
        headers["server-timing"] = timing
    if streaming:
        headers.update(SSE_HEADERS)
    return headers
