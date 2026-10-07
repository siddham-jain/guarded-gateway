"""a cached entry as json or as a synthesized sse stream, whichever way the entry was produced"""

from gg.cache.base import CachedResponse
from gg.core.context import RequestContext
from gg.core.ids import completion_id
from gg.core.schema import AssistantMessage, ChatResponse, Choice
from gg.pipeline.stage import PipelineResult, ResultSource
from gg.pipeline.streams import synthesize_stream


def to_response(entry: CachedResponse, *, request_id: str, created: int) -> ChatResponse:
    message = AssistantMessage(content=entry.content, refusal=entry.refusal)
    return ChatResponse(
        id=completion_id(request_id),
        created=created,
        model=entry.response_model,
        choices=(Choice(index=0, message=message, finish_reason=entry.finish_reason),),
        usage=entry.usage,
        system_fingerprint=entry.system_fingerprint,
    )


def to_result(
    entry: CachedResponse, ctx: RequestContext, source: ResultSource, created: int
) -> PipelineResult:
    response = to_response(entry, request_id=ctx.request_id, created=created)
    if ctx.request.stream:
        stream = synthesize_stream(response, include_usage=ctx.original.wants_usage())
        return PipelineResult(source=source, stream=stream)
    return PipelineResult(source=source, response=response)
