from typing import Annotated, Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from gg.core.schema import StrictModel

type Probability = Annotated[float, Field(ge=0, le=1)]
type Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class JevStateConfig(StrictModel):
    request_max_chars: PositiveInt = 4000
    request_head_chars: PositiveInt = 3000
    request_tail_chars: PositiveInt = 900
    prior_user_turns: Annotated[int, Field(ge=0)] = 2
    prior_user_turn_max_chars: PositiveInt = 600
    followup_max_words: PositiveInt = 30
    last_assistant_max_chars: PositiveInt = 800
    include_system_excerpt: bool = True
    system_excerpt_max_chars: PositiveInt = 500
    new_max_turns: PositiveInt = 1
    short_max_turns: PositiveInt = 5

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.request_head_chars + self.request_tail_chars >= self.request_max_chars:
            raise ValueError("request_head_chars + request_tail_chars must be below request_max_chars")
        if self.new_max_turns >= self.short_max_turns:
            raise ValueError("new_max_turns must be below short_max_turns")
        return self


class JevBreakerConfig(StrictModel):
    consecutive_failures: PositiveInt = 3
    cooldown_s: PositiveFloat = 30
    auth_cooldown_s: PositiveFloat = 300


class JevConfig(StrictModel):
    url: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-1.13.0"
    question_set: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")] = "qset-v1"
    state_builder: Literal["state-v1"] = "state-v1"
    # tuples, not sets: set order is hash-randomised and would make config_hash unstable
    strong_tiers: tuple[str, ...] = ("frontier", "frontier_reasoning")
    # strong_helps ranked strong-wins best in the routing eval (heldout auroc 0.81 vs 0.66 for tier);
    # tier = P(strong_tiers) is coarse: 0 for 107 of 170 eval prompts and never above 0.66
    score_signal: Literal["strong_helps", "tier"] = "strong_helps"
    price_per_mtok_input_usd: Annotated[float, Field(ge=0)] = 0.042
    attempt_timeout_ms: PositiveInt = 400
    connect_timeout_ms: PositiveInt = 300
    max_attempts: Annotated[int, Field(ge=1, le=2)] = 2
    min_attempt_budget_ms: PositiveInt = 150
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504, 529)
    retry_backoff_ms: tuple[PositiveInt, PositiveInt] = (100, 150)
    breaker: JevBreakerConfig = JevBreakerConfig()
    state: JevStateConfig = JevStateConfig()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.model.endswith("latest"):
            raise ValueError("pin a jev model version; thresholds are only valid for the pinned model")
        low, high = self.retry_backoff_ms
        if low > high:
            raise ValueError("retry_backoff_ms must be [low, high]")
        return self


class StaticScorerConfig(StrictModel):
    score: Probability = 0.5


class ScorerConfig(StrictModel):
    type: str = "jev"
    deadline_ms: PositiveInt = 600
    jev: JevConfig = JevConfig()
    static: StaticScorerConfig = StaticScorerConfig()


class CacheConfig(StrictModel):
    enabled: bool = True
    lru_entries: PositiveInt = 10_000
    ttl_s: PositiveFloat = 86_400


class SessionConfig(StrictModel):
    reuse_tool_continuations: bool = True
    ratchet: bool = False
    ttl_s: PositiveFloat = 3_600
    max_entries: PositiveInt = 10_000


class RoutingClaimConfig(StrictModel):
    threshold: Probability = 0.5
    action: Literal["strong", "ignore"] = "strong"


class EffortConfig(StrictModel):
    mode: Literal["off", "fixed", "hinted"] = "fixed"
    weak: Effort = "none"
    strong: Effort = "low"


class ProfileConfig(StrictModel):
    weak_group: str
    strong_group: str
    threshold: Probability | None = None
    calibrated: bool = False


class PolicyConfig(StrictModel):
    threshold: Probability = 0.5
    default_profile: str = "default"
    fallback_route: Literal["strong", "weak"] = "strong"
    routing_claim: RoutingClaimConfig = RoutingClaimConfig()
    session: SessionConfig = SessionConfig()
    effort: EffortConfig = EffortConfig()


class HeadersConfig(StrictModel):
    expose_score: bool = True
    decimals: Annotated[int, Field(ge=0, le=6)] = 2


class RoutingConfig(StrictModel):
    version: Literal[1] = 1
    enabled: bool = True
    scorer: ScorerConfig = ScorerConfig()
    cache: CacheConfig = CacheConfig()
    policy: PolicyConfig = PolicyConfig()
    profiles: dict[str, ProfileConfig] = Field(
        default_factory=lambda: {"default": ProfileConfig(weak_group="weak", strong_group="strong")}
    )
    headers: HeadersConfig = HeadersConfig()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.policy.default_profile not in self.profiles:
            raise ValueError(f"policy.default_profile '{self.policy.default_profile}' is not in profiles")
        return self
