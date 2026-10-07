"""promptguard.co guard api as a remote detector (input tier 2, output post-hoc).

only scrubbed text leaves the gateway: vault values are swapped back to their placeholders first, the
same rule as for the jev router. promptguard's own redaction is irreversible, so a `redact` decision is
reported as a flag and gg's reversible vault stays in charge of pii.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import httpx2
from pydantic import Field, field_validator

from gg.core.clock import Clock
from gg.core.guard_types import Verdict
from gg.core.jsonutil import loads
from gg.core.schema import Role, StrictModel
from gg.guardrails.base import GuardBackendError, GuardContext, GuardFinding, GuardStage, Streaming, finding
from gg.guardrails.registry import GuardDeps
from gg.guardrails.segments import scoped_texts
from gg.guardrails.vault import GuardVault

DISABLED = "promptguard_disabled"
API_BASE = "https://api.promptguard.co"
GUARD_PATH = "/api/v1/guard"


class PromptGuardError(GuardBackendError):
    """the api could not decide; the engine applies the guard's on_error policy"""


class PromptGuardCfg(StrictModel):
    api_base: str = API_BASE
    roles: tuple[Role, ...] = ("user", "tool")
    history_user_turns: Annotated[int, Field(ge=1, le=100)] = 3
    # threat types that block; anything else promptguard blocks on is downgraded to a flag
    block_threats: tuple[str, ...] = (
        "prompt_injection",
        "system_prompt_leak",
        "data_exfiltration",
        "multi_turn_escalation",
        "malware",
        "api_key_leak",
        "secret_key_leak",
    )
    # never send more than this per call; long prompts keep head and tail
    max_chars: Annotated[int, Field(ge=1_000, le=100_000)] = 16_000
    # after this many failures in a row the api is skipped for open_s, so an outage costs no latency
    breaker_failures: Annotated[int, Field(ge=1)] = 3
    breaker_open_s: Annotated[float, Field(gt=0)] = 60.0
    # below the policy's timeout_ms, so a slow api fails here and counts towards the breaker
    request_timeout_s: Annotated[float, Field(gt=0, le=30)] = 0.9

    @field_validator("roles")
    @classmethod
    def _no_operator_roles(cls, value: tuple[Role, ...]) -> tuple[Role, ...]:
        if {"system", "developer"} & set(value):
            raise ValueError("system and developer text is operator-controlled and never sent out")
        return value


def scrub(text: str, vault: GuardVault) -> str:
    # longest values first so a value contained in another is not replaced inside it
    for placeholder, value in sorted(vault.placeholders().items(), key=lambda kv: -len(kv[1])):
        text = text.replace(value, placeholder)
    return text


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n…\n" + text[-half:]


def verdict_of(body: Mapping[str, Any], block_threats: frozenset[str]) -> tuple[Verdict, str, float | None]:
    decision = body.get("decision")
    threat = body.get("threat_type") or ""
    score = body.get("weighted_score") or body.get("confidence")
    score = float(score) if isinstance(score, (int, float)) else None
    if decision == "block":
        return (Verdict.BLOCK if threat in block_threats else Verdict.FLAG), threat or "blocked", score
    if decision == "redact":
        return Verdict.FLAG, threat or "redact", score
    if decision == "allow":
        return Verdict.ALLOW, "", score
    raise PromptGuardError(f"unexpected decision {decision!r}")


class PromptGuard:
    tier: int = 2

    def __init__(
        self,
        cfg: PromptGuardCfg,
        client: httpx2.AsyncClient | None,
        api_key: str | None,
        *,
        stage: GuardStage,
        clock: Clock,
    ) -> None:
        self._cfg = cfg
        self._clock = clock
        self._failures = 0
        self._open_until = 0.0
        self._client = client
        self._api_key = api_key
        self._roles = frozenset(cfg.roles)
        self._block = frozenset(cfg.block_threats)
        self.stage: GuardStage = stage
        self.name: str = "promptguard" if stage == "input" else "promptguard_output"
        self.streaming: Streaming = "windowed" if stage == "input" else "post_hoc"
        if stage == "output":
            self.tier = 3

    @property
    def available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        client, api_key = self._client, self._api_key
        if client is None or not api_key:
            return finding(self.name, self.stage, reason=DISABLED)
        messages = self._messages(gctx)
        if not messages:
            return finding(self.name, self.stage)
        if self._clock.monotonic() < self._open_until:
            raise PromptGuardError("circuit open after repeated failures")
        try:
            body = await self._call(client, api_key, messages, direction=self.stage, model=gctx.request.model)
        except PromptGuardError:
            self._failures += 1
            if self._failures >= self._cfg.breaker_failures:
                self._open_until = self._clock.monotonic() + self._cfg.breaker_open_s
                self._failures = 0
            raise
        self._failures = 0
        verdict, reason, score = verdict_of(body, self._block)
        labels = tuple(sorted({str(t.get("type")) for t in body.get("threats", []) if isinstance(t, dict)}))
        return finding(self.name, self.stage, verdict, score=score, reason=reason, labels=labels)

    def _messages(self, gctx: GuardContext) -> list[dict[str, str]]:
        if self.stage == "input":
            texts: Sequence[str] = scoped_texts(gctx.segments, self._roles, self._cfg.history_user_turns)
            role = "user"
        else:
            texts = [seg.text for seg in gctx.segments if seg.text.strip()]
            role = "assistant"
        joined = "\n\n".join(scrub(t, gctx.vault) for t in texts)
        return [{"role": role, "content": clip(joined, self._cfg.max_chars)}] if joined.strip() else []

    async def _call(
        self,
        client: httpx2.AsyncClient,
        api_key: str,
        messages: list[dict[str, str]],
        *,
        direction: str,
        model: str,
    ) -> Mapping[str, Any]:
        try:
            response = await client.post(
                self._cfg.api_base.rstrip("/") + GUARD_PATH,
                json={"messages": messages, "direction": direction, "model": model},
                headers={"X-API-Key": api_key},
                timeout=self._cfg.request_timeout_s,
            )
        except httpx2.HTTPError as exc:
            raise PromptGuardError(f"transport: {type(exc).__name__}") from exc
        if response.status_code == 429:
            raise PromptGuardError("quota or rate limit exceeded")
        if response.status_code in (401, 403):
            raise PromptGuardError("invalid promptguard api key")
        if response.status_code != 200:
            raise PromptGuardError(f"status {response.status_code}")
        body = loads(response.content)
        if not isinstance(body, dict):
            raise PromptGuardError("response is not an object")
        return body  # pyright: ignore[reportUnknownVariableType]


def _client(deps: GuardDeps, cfg: PromptGuardCfg) -> tuple[httpx2.AsyncClient | None, str | None]:
    remote = deps.remote
    if remote is None:
        return None, None
    return remote.client("promptguard", cfg.api_base), remote.api_keys.get("promptguard")


def create_input(cfg: PromptGuardCfg, deps: GuardDeps) -> PromptGuard:
    client, key = _client(deps, cfg)
    return PromptGuard(cfg, client, key, stage="input", clock=deps.clock)


def create_output(cfg: PromptGuardCfg, deps: GuardDeps) -> PromptGuard:
    client, key = _client(deps, cfg)
    return PromptGuard(cfg, client, key, stage="output", clock=deps.clock)
