import asyncio

import httpx2
import pytest

from gg.core.jsonutil import loads
from gg.observability.otlp import OtlpExporter, SpanData, encode_span

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
URL = "http://langfuse.test/api/public/otel/v1/traces"


def span(name: str = "s", **kwargs: object) -> SpanData:
    return SpanData(name=name, span_id="b7ad6b7169203331", start_ns=10, end_ns=20, **kwargs)  # pyright: ignore[reportArgumentType]


def test_encode_span_follows_the_otlp_json_mapping() -> None:
    encoded = encode_span(
        TRACE_ID,
        span(
            parent_id="00f067aa0ba902b7",
            kind="client",
            attributes={"s": "x", "b": True, "i": 3, "f": 0.5, "a": ("p", "q")},
            error="overloaded",
        ),
    )
    assert encoded == {
        "traceId": TRACE_ID,
        "spanId": "b7ad6b7169203331",
        "parentSpanId": "00f067aa0ba902b7",
        "name": "s",
        "kind": 3,
        "startTimeUnixNano": "10",
        "endTimeUnixNano": "20",
        "attributes": [
            {"key": "s", "value": {"stringValue": "x"}},
            {"key": "b", "value": {"boolValue": True}},
            {"key": "i", "value": {"intValue": "3"}},
            {"key": "f", "value": {"doubleValue": 0.5}},
            {"key": "a", "value": {"arrayValue": {"values": [{"stringValue": "p"}, {"stringValue": "q"}]}}},
        ],
        "status": {"code": 2, "message": "overloaded"},
    }


def test_end_never_precedes_start() -> None:
    assert encode_span(TRACE_ID, SpanData("s", "1" * 16, 50, 40))["endTimeUnixNano"] == "50"


class Upstream:
    def __init__(self, status: int = 200, fail: bool = False) -> None:
        self.status = status
        self.fail = fail
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        if self.fail:
            raise httpx2.ConnectError("refused", request=request)
        self.requests.append(request)
        return httpx2.Response(self.status, json={})

    def spans(self) -> list[str]:
        out: list[str] = []
        for r in self.requests:
            body = loads(r.content)
            out += [s["name"] for s in body["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        return out


def exporter(upstream: Upstream, dropped: list[int], **kwargs: object) -> OtlpExporter:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(upstream))
    return OtlpExporter(
        client,
        URL,
        headers={"authorization": "Basic abc"},
        resource={"service.name": "gg-gateway"},
        on_dropped=dropped.append,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


def trace(name: str) -> list[dict[str, object]]:
    return [encode_span(TRACE_ID, span(name))]


async def test_background_task_posts_batches_with_auth() -> None:
    upstream, dropped = Upstream(), list[int]()
    exp = exporter(upstream, dropped, flush_interval_s=0.01)
    await exp.start()
    exp.submit(trace("a"))
    for _ in range(100):
        if upstream.requests:
            break
        await asyncio.sleep(0.01)
    await exp.stop()
    request = upstream.requests[0]
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Basic abc"
    assert request.headers["content-type"] == "application/json"
    body = loads(request.content)
    assert body["resourceSpans"][0]["resource"]["attributes"] == [
        {"key": "service.name", "value": {"stringValue": "gg-gateway"}}
    ]
    assert upstream.spans() == ["a"]
    assert sum(dropped) == 0


async def test_stop_flushes_the_queue_in_batches() -> None:
    upstream, dropped = Upstream(), list[int]()
    exp = exporter(upstream, dropped, batch_max=2)
    for name in "abcde":
        exp.submit(trace(name))
    await exp.stop()
    assert len(upstream.requests) == 3
    assert upstream.spans() == list("abcde")


async def test_full_queue_drops_instead_of_blocking() -> None:
    upstream, dropped = Upstream(), list[int]()
    exp = exporter(upstream, dropped, queue_max=1)
    assert exp.submit(trace("a"))
    assert not exp.submit(trace("b"))
    assert dropped == [1]


@pytest.mark.parametrize("upstream", [Upstream(status=401), Upstream(fail=True)])
async def test_failed_exports_are_counted_not_raised(upstream: Upstream) -> None:
    dropped: list[int] = []
    exp = exporter(upstream, dropped)
    exp.submit(trace("a"))
    exp.submit(trace("b"))
    await exp.stop()
    assert sum(dropped) == 2
