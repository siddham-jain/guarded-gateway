import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx2

from gg.core.clock import FakeClock
from gg.core.deployment import Capabilities, Deployment
from gg.core.schema import ChatRequest, Message
from gg.providers.base import AliasRef, DeploymentRef, PublicModel, Resolution
from gg.routing.base import RoutingRequest
from gg.routing.config import JevConfig
from gg.routing.scorers.jev.client import JevClient
from gg.routing.scorers.jev.questions import QuestionSet, load_question_set
from gg.routing.scorers.jev.scorer import JevScorer
from gg.routing.scorers.jev.state import StateBuilder

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "config"
FIXTURES = ROOT / "tests/fixtures/jev"

type Handler = Callable[[httpx2.Request], httpx2.Response | Awaitable[httpx2.Response]]


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def all_fixtures() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("p*.json"))


def qset(cfg: JevConfig | None = None) -> QuestionSet:
    return load_question_set(CONFIG_DIR, "qset-v1", (cfg or JevConfig()).strong_tiers)


def routing_request(*messages: dict[str, Any], tools: bool = False) -> RoutingRequest:
    return RoutingRequest(
        request_id="req_test",
        key_id="test-key",
        messages=tuple(Message.model_validate(m) for m in messages),
        tools_present=tools,
    )


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def json_response(body: dict[str, Any], status: int = 200, **headers: str) -> httpx2.Response:
    return httpx2.Response(status, json=body, headers=headers)


class FakeSleep:
    """advances the fake clock instead of sleeping, and records each delay"""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self._clock.advance(seconds)


def make_client(
    handler: Handler,
    clock: FakeClock,
    cfg: JevConfig | None = None,
    *,
    sleep: FakeSleep | None = None,
    api_key: str = "jev-test-key",
) -> JevClient:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return JevClient(http, api_key, cfg or JevConfig(), clock, sleep=sleep or FakeSleep(clock))


def make_scorer(client: JevClient, cfg: JevConfig | None = None, *, deadline_s: float = 0.6) -> JevScorer:
    cfg = cfg or JevConfig()
    return JevScorer(client, StateBuilder(cfg.state), qset(cfg), cfg, deadline_s=deadline_s)


def _dep(dep_id: str, **caps: Any) -> Deployment:
    provider, _, model = dep_id.partition("/")
    return Deployment(id=dep_id, provider=provider, upstream_model=model, capabilities=Capabilities(**caps))


WEAK = _dep("openai/gpt-6-luna", effort_levels=frozenset({"none", "low", "medium", "high"}))
HAIKU = _dep("anthropic/claude-haiku-4-5")
STRONG = _dep("openai/gpt-6.1-sol", effort_levels=frozenset({"low", "medium", "high"}))
STRONG_TOOLS = _dep("openai/gpt-6-sol", tools_require_effort="none", effort_levels=frozenset({"none", "low"}))
SONNET = _dep("anthropic/claude-sonnet-5-5")


class FakeCatalog:
    hash = "test"
    group_names = ("weak", "strong")

    def __init__(self, chains: dict[str, list[Deployment]] | None = None) -> None:
        self._chains = chains or {}

    def resolve(self, model: str, /) -> Resolution | None:
        if model == "gg/auto":
            return AliasRef("gg/auto", "router")
        if model in ("gg/weak", "gg/strong"):
            return AliasRef(model, "group")
        for dep in (WEAK, HAIKU, STRONG, STRONG_TOOLS, SONNET):
            if model == dep.id:
                return DeploymentRef(dep)
        return None

    def get(self, deployment_id: str, /) -> Deployment:
        raise KeyError(deployment_id)

    def chain(self, group: str, request: ChatRequest, /) -> list[Deployment]:
        if group in self._chains:
            return self._chains[group]
        if group == "weak":
            return [WEAK, HAIKU]
        if group == "strong":
            return [STRONG_TOOLS, SONNET] if request.has_tools() else [STRONG, SONNET]
        return []

    def groups_of(self, alias: str, /) -> tuple[str, ...]:
        return ("weak", "strong") if alias == "gg/auto" else ()

    def deployments_for(self, model: str, /) -> list[Deployment]:
        return []

    def list_public(self, allowed: Callable[[str], bool], /) -> Sequence[PublicModel]:
        return []
