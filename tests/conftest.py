from collections.abc import Iterator
from datetime import date
from typing import Any

import pytest
import structlog

from gg.core.aio import Deadline
from gg.core.clock import FakeClock
from gg.core.context import RequestContext, StageTimings
from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    # build_app configures structlog process-wide; keep one test's level from hiding another's logs
    yield
    structlog.reset_defaults()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_request(**overrides: Any) -> ChatRequest:
    payload: dict[str, Any] = {
        "model": "gg/auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


def make_key(**overrides: Any) -> KeyPolicy:
    payload: dict[str, Any] = {
        "id": "test-key",
        "name": "Test",
        "prefix": "gg-test-abcd",
        "created_at": date(2026, 10, 5),
    }
    payload.update(overrides)
    return KeyPolicy.model_validate(payload)


def make_ctx(clock: FakeClock, request: ChatRequest | None = None) -> RequestContext:
    req = request or make_request()
    return RequestContext(
        request_id="req_test",
        received_at=clock.monotonic(),
        received_unix=int(clock.time()),
        key=make_key(),
        original=req,
        request=req,
        deadline=Deadline.after(30, clock),
        timings=StageTimings(clock),
    )
