from typing import Any

import httpx2
import pytest

from gg.cache.config import VerifierConfig
from gg.cache.verifiers import build_verifier
from gg.cache.verifiers.jev import JevPairVerifier, VerifierError
from gg.core.jsonutil import loads


def verifier(handler: Any, **cfg: Any) -> JevPairVerifier:
    client = httpx2.AsyncClient(base_url="https://jev.test", transport=httpx2.MockTransport(handler))
    return JevPairVerifier(client, "jev-key", VerifierConfig(type="jev", **cfg))


def answering(score: Any, status: int = 200) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        handler.requests.append(request)  # type: ignore[attr-defined]
        body = {"answers": {"same_answer": {"type": "noul", "noul": score}}}
        return httpx2.Response(status, json=body)

    handler.requests = []  # type: ignore[attr-defined]
    return handler


@pytest.mark.parametrize(("score", "same"), [(0.95, True), (0.8, True), (0.79, False), (0.02, False)])
async def test_score_is_compared_with_min_score(score: float, same: bool) -> None:
    assert await verifier(answering(score)).same_answer("a", "b") is same


async def test_request_carries_both_prompts_one_question_and_the_key() -> None:
    handler = answering(0.9)
    await verifier(handler).same_answer("capital of France?", "France's capital?")
    request = handler.requests[0]
    body = loads(request.content)
    assert request.url.path == "/v1/systemone"
    assert request.headers["authorization"] == "Bearer jev-key"
    assert body["state"] == {"cached_prompt": "capital of France?", "new_prompt": "France's capital?"}
    assert list(body["questions"]) == ["same_answer"]


@pytest.mark.parametrize("handler", [answering(0.9, status=503), answering("yes"), answering(True)])
async def test_anything_but_a_numeric_score_is_an_error(handler: Any) -> None:
    with pytest.raises(VerifierError):
        await verifier(handler).same_answer("a", "b")


async def test_a_slow_answer_is_an_error() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(VerifierError):
        await verifier(slow).same_answer("a", "b")


def test_builder_returns_none_without_a_type_or_a_key() -> None:
    client = httpx2.AsyncClient()
    assert build_verifier(VerifierConfig(), lambda _: client, "key") is None
    assert build_verifier(VerifierConfig(type="jev"), lambda _: client, None) is None
    assert isinstance(build_verifier(VerifierConfig(type="jev"), lambda _: client, "key"), JevPairVerifier)
