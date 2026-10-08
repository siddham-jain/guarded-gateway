"""gg/auto through the real composition root: ci profile, mock providers, mocked jev transport"""

import asyncio
import json
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx2
import openai
import yaml
from fastapi import FastAPI

from gg.app.factory import Overrides, build_app
from gg.config.settings import Settings
from gg.core.clock import SystemClock
from gg.limits.spend_guard import InMemorySpendGuard, SpendGuardSettings
from tests.unit.routing.support import ROOT, fixture

TOKENS: dict[str, str] = yaml.safe_load((ROOT / "tests/fixtures/keys/tokens.yaml").read_text())
MESSAGES: Any = [
    {"role": "user", "content": "Design a rate limiter for a multi-region gateway. Reply to [EMAIL_1]."}
]

type JevHandler = Callable[[httpx2.Request], Awaitable[httpx2.Response]]


def jev_body(strong_mass: float) -> dict[str, Any]:
    body = fixture("p14")["response"]
    body["answers"]["tier"]["probabilities"] = {
        "small": 0.0,
        "mid": round(1 - strong_mass, 4),
        "frontier": strong_mass,
        "frontier_reasoning": 0.0,
    }
    body["answers"]["strong_helps"] = {"type": "noul", "noul": strong_mass}
    return body


def config_dir(tmp_path: Path, threshold: float) -> Path:
    target = tmp_path / f"config-{threshold}"
    shutil.copytree(ROOT / "config", target)
    routing = yaml.safe_load((target / "routing.yaml").read_text())
    routing["policy"]["threshold"] = threshold
    (target / "routing.yaml").write_text(yaml.safe_dump(routing))
    return target


@asynccontextmanager
async def gg_client(cfg_dir: Path, jev: JevHandler) -> AsyncIterator[openai.AsyncOpenAI]:
    settings = Settings(  # pyright: ignore[reportCallIssue]
        _env_file=None,
        env="test",
        log_format="console",
        log_level="warning",
        config_dir=cfg_dir,
        keys_file=ROOT / "tests/fixtures/keys/keys.yaml",
        model_profile="ci",
        jev_api_key="jev-test-key",
    )
    caps = SpendGuardSettings(  # pyright: ignore[reportCallIssue]
        _env_file=None, cap_daily_usd="1.00", cap_total_usd="1.00", cap_run_usd="1.00"
    )
    overrides = Overrides(
        transports={"jev": httpx2.MockTransport(jev)},
        spend_guard=InMemorySpendGuard(caps, clock=SystemClock()),
    )
    app: FastAPI = build_app(settings, overrides=overrides)
    async with app.router.lifespan_context(app):
        yield openai.AsyncOpenAI(
            base_url="http://gg.test/v1",
            api_key=TOKENS["demo"],
            http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
            max_retries=0,
        )


class MockJev:
    def __init__(self, strong_mass: float = 0.6, delay_s: float = 0.0) -> None:
        self._body = jev_body(strong_mass)
        self._delay = delay_s
        self.requests: list[httpx2.Request] = []
        self.other: list[httpx2.Request] = []

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        questions = json.loads(request.content)["questions"]
        if "tier" not in questions:
            # the injection guard and the cache verifier ask jev too; answer "no" to every question
            self.other.append(request)
            answers = {name: {"type": "noul", "noul": 0.0} for name in questions}
            return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": answers})
        self.requests.append(request)
        await asyncio.sleep(self._delay)
        return httpx2.Response(200, json=self._body, headers={"x-typesafe-request-id": "req_mock"})


async def route(cfg_dir: Path, jev: MockJev) -> httpx2.Headers:
    async with gg_client(cfg_dir, jev) as oai:
        raw = await oai.chat.completions.with_raw_response.create(model="gg/auto", messages=MESSAGES)
        raw.parse()
        return raw.headers


async def test_threshold_moves_the_same_score_between_tiers(tmp_path: Path) -> None:
    jev = MockJev(strong_mass=0.6)
    quality = await route(config_dir(tmp_path, 0.3), jev)
    economy = await route(config_dir(tmp_path, 0.9), jev)

    assert (quality["x-gg-route"], quality["x-gg-route-reason"]) == ("strong", "scored")
    assert (economy["x-gg-route"], economy["x-gg-route-reason"]) == ("weak", "scored")
    assert quality["x-gg-route-score"] == economy["x-gg-route-score"] == "0.60"
    assert (quality["x-gg-route-threshold"], economy["x-gg-route-threshold"]) == ("0.30", "0.90")
    assert quality["x-gg-config"] != economy["x-gg-config"]

    sent = jev.requests[0]
    assert sent.headers["authorization"] == "Bearer jev-test-key"
    body = json.loads(sent.content)
    assert body["model"] == "jev-1.13.0"
    assert body["state"]["request"] == MESSAGES[0]["content"]
    assert "tier" in body["questions"]


async def test_repeat_prompt_is_answered_from_the_score_cache(tmp_path: Path) -> None:
    jev = MockJev(strong_mass=0.9)
    async with gg_client(config_dir(tmp_path, 0.5), jev) as oai:
        for _ in range(2):
            raw = await oai.chat.completions.with_raw_response.create(model="gg/auto", messages=MESSAGES)
            assert raw.headers["x-gg-route"] == "strong"
    assert len(jev.requests) == 1


async def test_jev_timeout_fails_open_to_strong(tmp_path: Path) -> None:
    jev = MockJev(strong_mass=0.0, delay_s=2.0)
    async with asyncio.timeout(1.5):
        headers = await route(config_dir(tmp_path, 0.5), jev)
    assert headers["x-gg-route"] == "strong"
    assert headers["x-gg-route-reason"] == "fallback:timeout"
    assert "x-gg-route-score" not in headers


async def test_non_router_models_never_ask_jev_to_route(tmp_path: Path) -> None:
    jev = MockJev()
    async with gg_client(config_dir(tmp_path, 0.5), jev) as oai:
        raw = await oai.chat.completions.with_raw_response.create(model="gg/weak", messages=MESSAGES)
    assert raw.headers["x-gg-route-reason"] == "alias"
    assert jev.requests == []
    assert [list(json.loads(r.content)["questions"]) for r in jev.other] == [["injection"]]
