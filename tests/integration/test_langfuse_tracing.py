"""requests through the real app produce one langfuse trace each, flushed on shutdown"""

from typing import Any

import httpx2
import pytest
from fastapi import FastAPI
from mock_upstream.app import app as mock_upstream_app
from tests.integration.test_walking_skeleton import client, settings, spend_guard

from gg.app.factory import Overrides, build_app
from gg.core.jsonutil import loads

EMAIL = "jane@corp.io"
MESSAGES: Any = [{"role": "user", "content": f"say hi to {EMAIL}"}]


class Langfuse:
    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return httpx2.Response(200, json={})

    def traces(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for r in self.requests:
            for s in loads(r.content)["resourceSpans"][0]["scopeSpans"][0]["spans"]:
                out.setdefault(s["traceId"], []).append(s)
        return out


def attrs(span: dict[str, Any]) -> dict[str, Any]:
    return {a["key"]: next(iter(a["value"].values())) for a in span["attributes"]}


@pytest.fixture
def langfuse() -> Langfuse:
    return Langfuse()


@pytest.fixture
def app(langfuse: Langfuse) -> FastAPI:
    overrides = Overrides(
        transports={
            "mock_http": httpx2.ASGITransport(app=mock_upstream_app),
            "langfuse": httpx2.MockTransport(langfuse),
        },
        spend_guard=spend_guard(),
    )
    return build_app(
        settings(
            "loadtest",
            providers={"mock_http": {"base_url": "http://mock-upstream/v1"}},
            langfuse={
                "host": "http://langfuse.test",
                "public_key": "pk-lf-test",
                "secret_key": "sk-lf-test",
                "capture_content": "scrubbed",
            },
        ),
        overrides=overrides,
    )


async def test_each_request_becomes_one_trace(app: FastAPI, langfuse: Langfuse) -> None:
    gg = client(app)
    # leaving the lifespan flushes the exporter
    async with app.router.lifespan_context(app):
        plain = await gg.chat.completions.with_raw_response.create(model="gg/weak", messages=MESSAGES)
        streamed = await gg.chat.completions.with_raw_response.create(
            model="gg/weak", messages=MESSAGES, stream=True
        )
        async for _ in streamed.parse():
            pass
    trace_ids = [plain.headers["x-gg-trace-id"], streamed.headers["x-gg-trace-id"]]
    assert plain.parse().choices[0].message.content == f"say hi to {EMAIL}"
    traces = langfuse.traces()
    assert sorted(traces) == sorted(trace_ids)
    assert langfuse.requests[0].url.path == "/api/public/otel/v1/traces"

    for trace_id in trace_ids:
        spans = {s["name"]: s for s in traces[trace_id]}
        root = attrs(spans["chat.completions"])
        assert root["langfuse.trace.metadata.status"] == "200"
        assert root["langfuse.trace.metadata.served_by"] == "mock_http/gpt-mock"
        generation = attrs(spans["chat gpt-mock"])
        assert generation["langfuse.observation.type"] == "generation"
        assert "langfuse.observation.usage_details" in generation
        assert {"gg.limits", "gg.guard_in_pre", "gg.guard_out"} <= set(spans)
        # only placeholder-space text leaves the gateway
        assert EMAIL not in str(traces[trace_id])
        assert "[EMAIL_1]" in root["langfuse.observation.input"]
        assert "[EMAIL_1]" in root["langfuse.observation.output"]
