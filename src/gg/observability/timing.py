"""request timing math (C10 §4.3) over StageTimings marks and the attempt log.

marks read (monotonic seconds, first write wins):
  received              asgi entry; falls back to ctx.received_at
  upstream_start        first upstream attempt starts
  upstream_first_token  first upstream content delta
  upstream_end          upstream body/stream finished or failed
  client_first_byte     first content byte written to the client (streams)
  completed             last byte written; set by the metrics finalizer if the writer did not
when ctx.attempts is non-empty its started_at/duration_s/ttft_s (same monotonic clock) win over the
upstream_* marks, because marks only keep the first attempt's values.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from gg.core.context import RequestContext
from gg.core.usage import AttemptRecord

CACHE_HITS = frozenset({"exact_hit", "semantic_hit"})


@dataclass(frozen=True, slots=True)
class RequestTimings:
    """seconds; None where a phase did not happen (no upstream, not a stream, ...)"""

    total: float
    pre_upstream: float | None = None
    upstream: float | None = None
    upstream_ttft: float | None = None
    failover: float = 0.0
    ttft: float | None = None
    ttft_added: float | None = None
    post: float | None = None
    stream_tail: float | None = None
    overhead: float | None = None
    tpot: float | None = None

    def overhead_phases(self) -> dict[str, float]:
        phases = {
            "pre_upstream": self.pre_upstream,
            "ttft_added": self.ttft_added,
            "post": self.post,
            "stream_tail": self.stream_tail,
            "total": self.overhead,
        }
        result = {name: value for name, value in phases.items() if value is not None}
        if self.failover > 0:
            result["failover"] = self.failover
        return result


@dataclass(frozen=True, slots=True)
class _UpstreamSpan:
    first_start: float
    final_start: float
    final_first: float | None
    final_end: float | None


def final_attempt(attempts: Sequence[AttemptRecord]) -> AttemptRecord | None:
    for attempt in reversed(attempts):
        if attempt.outcome == "ok":
            return attempt
    return attempts[-1] if attempts else None


def _upstream_span(ctx: RequestContext) -> _UpstreamSpan | None:
    final = final_attempt(ctx.attempts)
    if final is not None:
        return _UpstreamSpan(
            first_start=ctx.attempts[0].started_at,
            final_start=final.started_at,
            final_first=None if final.ttft_s is None else final.started_at + final.ttft_s,
            final_end=final.started_at + final.duration_s,
        )
    marks = ctx.timings.marks
    start = marks.get("upstream_start")
    if start is None:
        return None
    return _UpstreamSpan(start, start, marks.get("upstream_first_token"), marks.get("upstream_end"))


def _pos(seconds: float) -> float:
    return max(0.0, seconds)


def compute_timings(ctx: RequestContext, *, now: float) -> RequestTimings:
    marks = ctx.timings.marks
    received = marks.get("received", ctx.received_at)
    completed = marks.get("completed", now)
    total = _pos(completed - received)
    client_first = marks.get("client_first_byte")
    stream = ctx.request.stream
    ttft = _pos(client_first - received) if stream and client_first is not None else None
    span = None if ctx.cache_status in CACHE_HITS else _upstream_span(ctx)
    if span is None:
        return RequestTimings(total=total, ttft=ttft)

    upstream = None if span.final_end is None else _pos(span.final_end - span.final_start)
    upstream_ttft = None if span.final_first is None else _pos(span.final_first - span.final_start)
    failover = _pos(span.final_start - span.first_start)
    tail = None if span.final_end is None else _pos(completed - span.final_end)
    tpot = None
    output_tokens = ctx.usage.output_tokens if ctx.usage else 0
    if output_tokens >= 2 and span.final_first is not None and span.final_end is not None:
        tpot = _pos(span.final_end - span.final_first) / (output_tokens - 1)

    timings = RequestTimings(
        total=total,
        pre_upstream=_pos(span.first_start - received),
        upstream=upstream,
        upstream_ttft=upstream_ttft,
        failover=failover,
        ttft=ttft,
        tpot=tpot,
    )
    if stream:
        ttft_added = None
        if ttft is not None and upstream_ttft is not None:
            ttft_added = _pos(ttft - upstream_ttft - failover)
        overhead = None if ttft_added is None or tail is None else ttft_added + tail
        return replace(timings, ttft_added=ttft_added, stream_tail=tail, overhead=overhead)
    overhead = None if upstream is None else _pos(total - upstream - failover)
    return replace(timings, post=tail, overhead=overhead)
