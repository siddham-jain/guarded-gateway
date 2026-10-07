import asyncio
import json

import httpx2
import pytest

from gg.core.clock import FakeClock
from gg.routing.config import JevConfig
from gg.routing.scorers.jev.client import JevAnswer, JevFailure
from tests.unit.routing.support import (
    FakeSleep,
    fixture,
    json_response,
    make_client,
    make_scorer,
    routing_request,
    user,
)

BODY = b'{"model":"jev-1.13.0","state":{"request":"hi"},"questions":{}}'
OK = fixture("p13")["response"]


class Script:
    """mock jev that plays back one response (or exception) per call"""

    def __init__(self, *steps: httpx2.Response | Exception) -> None:
        self._steps = list(steps)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        step = self._steps.pop(0) if len(self._steps) > 1 else self._steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def ok(**headers: str) -> httpx2.Response:
    return json_response(OK, **({"x-typesafe-request-id": "req_abc"} | headers))


async def test_success_sends_auth_and_captures_request_id(clock: FakeClock) -> None:
    script = Script(ok())
    result = await make_client(script, clock).ask(BODY, 0.6)
    assert isinstance(result, JevAnswer)
    assert result.request_id == "req_abc"
    assert result.body["model"] == "jev-1.13.0"
    sent = script.requests[0]
    assert str(sent.url) == "https://api.typesafe.ai/v1/systemone"
    assert sent.headers["authorization"] == "Bearer jev-test-key"
    assert sent.headers["user-agent"].startswith("gg/")
    assert sent.content == BODY


async def test_scorer_sends_pinned_model_qset_and_scrubbed_state(clock: FakeClock) -> None:
    script = Script(ok())
    scorer = make_scorer(make_client(script, clock))
    score = await scorer.score(routing_request(user("Why do my Celery workers hang?")))
    assert not score.fallback
    assert score.request_id == "req_abc"
    assert score.scorer_version == "jev-1.13.0:qset-v1:state-v1:strong_helps" == scorer.version
    sent = json.loads(script.requests[0].content)
    assert sent["model"] == "jev-1.13.0"
    assert sent["state"]["request"] == "Why do my Celery workers hang?"
    assert list(sent["questions"]) == [
        "tier",
        "strong_helps",
        "difficulty",
        "task_type",
        "needs_multistep_reasoning",
        "routing_claim_present",
    ]


async def test_unscorable_request_makes_no_call(clock: FakeClock) -> None:
    script = Script(ok())
    score = await make_scorer(make_client(script, clock)).score(
        routing_request({"role": "system", "content": "x"})
    )
    assert score.fallback_reason == "unscorable"
    assert script.requests == []


async def test_invalid_answers_fall_back_with_parse_error(clock: FakeClock) -> None:
    script = Script(json_response({"model": "jev-1.13.0", "answers": {}}))
    score = await make_scorer(make_client(script, clock)).score(routing_request(user("hi")))
    assert score.fallback_reason == "parse_error"


async def test_attempt_timeout_is_not_retried(clock: FakeClock) -> None:
    async def slow(request: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(1)
        return ok()

    client = make_client(slow, clock, JevConfig(attempt_timeout_ms=30))
    result = await client.ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "timeout"
    assert client.breaker.state == "closed"


async def test_retries_5xx_once_after_jittered_backoff(clock: FakeClock) -> None:
    sleep = FakeSleep(clock)
    script = Script(httpx2.Response(503), ok())
    result = await make_client(script, clock, sleep=sleep).ask(BODY, 0.6)
    assert isinstance(result, JevAnswer)
    assert len(script.requests) == 2
    assert len(sleep.calls) == 1
    assert 0.1 <= sleep.calls[0] <= 0.15


@pytest.mark.parametrize(("status", "reason"), [(500, "http_5xx"), (529, "http_529"), (502, "http_5xx")])
async def test_gives_up_after_one_retry(clock: FakeClock, status: int, reason: str) -> None:
    script = Script(httpx2.Response(status))
    result = await make_client(script, clock).ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == reason
    assert len(script.requests) == 2


async def test_429_honours_retry_after_ms_when_it_fits(clock: FakeClock) -> None:
    sleep = FakeSleep(clock)
    script = Script(httpx2.Response(429, headers={"retry-after-ms": "40"}), ok())
    result = await make_client(script, clock, sleep=sleep).ask(BODY, 0.6)
    assert isinstance(result, JevAnswer)
    assert sleep.calls == [pytest.approx(0.04)]


async def test_429_with_long_retry_after_falls_back_without_retry(clock: FakeClock) -> None:
    script = Script(httpx2.Response(429, headers={"retry-after": "5"}))
    client = make_client(script, clock)
    result = await client.ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "http_429"
    assert len(script.requests) == 1
    assert client.breaker.state == "closed"


async def test_no_retry_when_remaining_budget_is_too_small(clock: FakeClock) -> None:
    calls: list[httpx2.Request] = []

    def slow_503(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        clock.advance(0.4)
        return httpx2.Response(503)

    result = await make_client(slow_503, clock).ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "http_5xx"
    assert len(calls) == 1


async def test_network_error_is_retried(clock: FakeClock) -> None:
    script = Script(httpx2.ConnectError("reset"), ok())
    result = await make_client(script, clock).ask(BODY, 0.6)
    assert isinstance(result, JevAnswer)
    assert len(script.requests) == 2


@pytest.mark.parametrize("status", [401, 402, 403])
async def test_auth_errors_open_the_breaker_for_five_minutes(clock: FakeClock, status: int) -> None:
    script = Script(json_response({"detail": "bad key"}, status=status))
    client = make_client(script, clock)
    first = await client.ask(BODY, 0.6)
    assert isinstance(first, JevFailure)
    assert first.reason == "auth"
    assert len(script.requests) == 1
    clock.advance(299)
    second = await client.ask(BODY, 0.6)
    assert isinstance(second, JevFailure)
    assert second.reason == "circuit_open"
    assert len(script.requests) == 1
    clock.advance(2)
    assert client.breaker.state == "half_open"


async def test_html_403_is_a_firewall_block_not_auth(clock: FakeClock) -> None:
    script = Script(httpx2.Response(403, text="<html>blocked</html>", headers={"content-type": "text/html"}))
    client = make_client(script, clock)
    result = await client.ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "firewall_403"
    assert client.breaker.state == "closed"


async def test_422_is_not_retried(clock: FakeClock) -> None:
    script = Script(json_response({"detail": [{"loc": ["questions"]}]}, status=422))
    result = await make_client(script, clock).ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "http_4xx"
    assert len(script.requests) == 1


async def test_garbage_200_is_a_parse_error(clock: FakeClock) -> None:
    script = Script(httpx2.Response(200, text="<html>oops"))
    result = await make_client(script, clock).ask(BODY, 0.6)
    assert isinstance(result, JevFailure)
    assert result.reason == "parse_error"


async def test_breaker_opens_after_three_consecutive_failures_and_recovers(clock: FakeClock) -> None:
    script = Script(httpx2.Response(500))
    client = make_client(script, clock, JevConfig(max_attempts=1))
    for _ in range(3):
        result = await client.ask(BODY, 0.6)
        assert isinstance(result, JevFailure)
        assert result.reason == "http_5xx"
    assert client.breaker.state == "open"
    blocked = await client.ask(BODY, 0.6)
    assert isinstance(blocked, JevFailure)
    assert blocked.reason == "circuit_open"
    assert len(script.requests) == 3

    clock.advance(30)
    assert client.breaker.state == "half_open"
    failed_probe = await client.ask(BODY, 0.6)
    assert isinstance(failed_probe, JevFailure)
    assert client.breaker.state == "open"

    clock.advance(30)
    healthy = make_client(Script(ok()), clock, JevConfig(max_attempts=1))
    healthy.breaker = client.breaker
    assert isinstance(await healthy.ask(BODY, 0.6), JevAnswer)
    assert client.breaker.state == "closed"


async def test_two_failures_then_success_keeps_breaker_closed(clock: FakeClock) -> None:
    script = Script(
        httpx2.Response(500), httpx2.Response(500), ok(), httpx2.Response(500), httpx2.Response(500)
    )
    client = make_client(script, clock, JevConfig(max_attempts=1))
    for _ in range(5):
        await client.ask(BODY, 0.6)
    assert client.breaker.state == "closed"


async def test_cancellation_propagates_and_releases_half_open_probe(clock: FakeClock) -> None:
    started = asyncio.Event()

    async def hang(request: httpx2.Request) -> httpx2.Response:
        started.set()
        await asyncio.sleep(10)
        return ok()

    client = make_client(hang, clock)
    client.breaker.trip(1)
    clock.advance(2)
    task = asyncio.create_task(client.ask(BODY, 0.6))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.breaker.allow()
