from starlette.requests import Request

from gg.api.authorize import AuthorizedRequest
from gg.api.deps import ApiServices
from gg.core.aio import Deadline
from gg.core.context import RequestContext, StageTimings
from gg.core.keypolicy import KeyPolicy
from gg.core.normalize import Normalization
from gg.core.schema import ChatRequest


def create_context(
    request: Request,
    services: ApiServices,
    *,
    key: KeyPolicy,
    original: ChatRequest,
    authorized: AuthorizedRequest,
    normalizations: tuple[Normalization, ...],
) -> RequestContext:
    timings: StageTimings = request.state.timings
    clock = services.clock
    return RequestContext(
        request_id=request.state.request_id,
        received_at=timings.marks.get("received", clock.monotonic()),
        received_unix=int(clock.time()),
        key=key,
        original=original,
        request=authorized.request,
        # the cap; the executor narrows it per plan (master §7)
        deadline=Deadline.after(services.settings.deadlines.max_s, clock),
        timings=timings,
        config_hash=services.config_hash,
        trace_id=request.state.trace_id,
        ignored_params=set(authorized.ignored_params),
        normalizations=normalizations,
    )
