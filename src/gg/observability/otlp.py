"""otlp/json span encoding and a bounded background exporter; backend-neutral (langfuse, jaeger, tempo)"""

import asyncio
import contextlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx2
import structlog

from gg.core.jsonutil import dumps

log = structlog.get_logger("gg.observability.otlp")

TRACES_PATH = "/v1/traces"

type AttrValue = str | bool | int | float | tuple[str, ...]
type SpanKind = Literal["internal", "server", "client"]
type EncodedSpan = dict[str, Any]

_KIND = {"internal": 1, "server": 2, "client": 3}
_STATUS_ERROR = 2


@dataclass(frozen=True, slots=True)
class SpanData:
    name: str
    span_id: str
    start_ns: int
    end_ns: int
    parent_id: str | None = None
    kind: SpanKind = "internal"
    attributes: Mapping[str, AttrValue] = field(default_factory=lambda: {})
    error: str | None = None


def _value(value: AttrValue) -> dict[str, Any]:
    # bool before int: bool is an int subclass
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"arrayValue": {"values": [{"stringValue": v} for v in value]}}


def attributes(values: Mapping[str, AttrValue]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in values.items()]


def encode_span(trace_id: str, span: SpanData) -> EncodedSpan:
    out: EncodedSpan = {
        "traceId": trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": _KIND[span.kind],
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(max(span.end_ns, span.start_ns)),
        "attributes": attributes(span.attributes),
    }
    if span.parent_id is not None:
        out["parentSpanId"] = span.parent_id
    if span.error is not None:
        out["status"] = {"code": _STATUS_ERROR, "message": span.error}
    return out


def payload(spans: Sequence[EncodedSpan], resource: Mapping[str, AttrValue], scope: str) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": attributes(resource)},
                "scopeSpans": [{"scope": {"name": scope}, "spans": list(spans)}],
            }
        ]
    }


class OtlpExporter:
    """bounded queue drained by one background task; submit never blocks, awaits or raises.

    a full queue or a failed post drops the trace and counts it, so a slow or dead backend costs the
    gateway nothing but lost traces.
    """

    def __init__(
        self,
        client: httpx2.AsyncClient,
        url: str,
        *,
        headers: Mapping[str, str],
        resource: Mapping[str, AttrValue],
        on_dropped: Callable[[int], None],
        scope: str = "gg",
        queue_max: int = 2_048,
        batch_max: int = 64,
        flush_interval_s: float = 1.0,
        timeout_s: float = 5.0,
    ) -> None:
        self._client = client
        self._url = url
        self._headers = {**headers, "content-type": "application/json"}
        self._resource = dict(resource)
        self._scope = scope
        self._on_dropped = on_dropped
        self._queue: asyncio.Queue[list[EncodedSpan]] = asyncio.Queue(queue_max)
        self._batch_max = batch_max
        self._flush_interval_s = flush_interval_s
        self._timeout = httpx2.Timeout(timeout_s)
        self._timeout_s = timeout_s
        self._task: asyncio.Task[None] | None = None

    def submit(self, trace: list[EncodedSpan]) -> bool:
        try:
            self._queue.put_nowait(trace)
        except asyncio.QueueFull:
            self._on_dropped(1)
            return False
        return True

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="otlp-exporter")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        try:
            async with asyncio.timeout(self._timeout_s):
                while batch := self._take(self._batch_max):
                    await self._post(batch)
        except TimeoutError:
            pass
        self._on_dropped(self._queue.qsize())

    async def _run(self) -> None:
        while True:
            try:
                first = await asyncio.wait_for(self._queue.get(), self._flush_interval_s)
            except TimeoutError:
                continue
            await self._post([first, *self._take(self._batch_max - 1)])

    def _take(self, limit: int) -> list[list[EncodedSpan]]:
        batch: list[list[EncodedSpan]] = []
        while not self._queue.empty() and len(batch) < limit:
            batch.append(self._queue.get_nowait())
        return batch

    async def _post(self, traces: list[list[EncodedSpan]]) -> None:
        body = dumps(payload([s for t in traces for s in t], self._resource, self._scope))
        try:
            response = await self._client.post(
                self._url, content=body, headers=self._headers, timeout=self._timeout
            )
        except httpx2.HTTPError as exc:
            log.warning("otlp.export_failed", error=type(exc).__name__, traces=len(traces))
            self._on_dropped(len(traces))
            return
        if response.status_code >= 300:
            log.warning("otlp.export_rejected", status=response.status_code, traces=len(traces))
            self._on_dropped(len(traces))
