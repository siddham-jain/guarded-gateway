from datetime import date
from decimal import Decimal
from fnmatch import fnmatchcase
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, PositiveInt, StringConstraints, model_validator

from gg.core.deployment import Deployment
from gg.core.schema import StrictModel


class RateLimitPolicy(StrictModel):
    rpm: PositiveInt | None = 60
    tpm: PositiveInt | None = 200_000
    max_concurrent: PositiveInt | None = 10


class BudgetPolicy(StrictModel):
    # config is usd decimals; the limits layer converts to integer micro-usd at load
    monthly_usd: Decimal | None = None
    daily_usd: Decimal | None = None
    soft_limit_pct: Annotated[float, Field(gt=0, lt=1)] = 0.8
    max_request_usd: Decimal | None = None
    fail_mode: Literal["closed", "open"] = "closed"


class RoutingOverrides(StrictModel):
    threshold: Annotated[float, Field(ge=0, le=1)] | None = None
    allow_request_threshold: bool = False
    threshold_bounds: tuple[float, float] = (0.0, 1.0)
    fallback_route: Literal["strong", "weak"] | None = None
    profile: str = "default"
    strong_pct: Annotated[float, Field(gt=0, lt=1)] | None = None
    jev_opt_out: bool = False
    guards_enabled: bool = False
    ratchet: bool = False
    expose_score_header: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        low, high = self.threshold_bounds
        if not 0 <= low <= high <= 1:
            raise ValueError("threshold_bounds must satisfy 0 <= low <= high <= 1")
        if self.threshold is not None and not low <= self.threshold <= high:
            raise ValueError("threshold must lie within threshold_bounds")
        return self


class CachePolicy(StrictModel):
    scope: Literal["key", "global", "off"] = "key"
    semantic: bool = False
    allow_sampled: bool = False
    default_ttl_s: PositiveInt = 86_400
    max_ttl_s: PositiveInt = 604_800


class GuardrailOverrides(StrictModel):
    policy_id: str = "default"
    allow_request_tightening: bool = True
    allow_paid_guards: bool = False


class RequestLimits(StrictModel):
    max_body_bytes: PositiveInt = 4 * 1024 * 1024
    max_messages: PositiveInt = 256
    max_tools: PositiveInt = 128
    max_completion_tokens: PositiveInt | None = 8192
    max_n: PositiveInt = 1


class KeyFlags(StrictModel):
    allow_free_tier_providers: bool = True
    allow_premium: bool = False
    allow_service_tier: bool = False
    allow_store: bool = False
    capture_content: bool = False
    trust_trace_context: bool = False


class KeyPolicy(StrictModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")]
    name: str
    prefix: str
    status: Literal["active", "disabled"] = "active"
    created_at: date
    expires_at: AwareDatetime | None = None
    tags: frozenset[str] = frozenset()
    allowed_models: tuple[str, ...] = ("gg/*",)
    allowed_providers: tuple[str, ...] | None = None
    rate_limits: RateLimitPolicy = RateLimitPolicy()
    budget: BudgetPolicy = BudgetPolicy()
    routing: RoutingOverrides = RoutingOverrides()
    cache: CachePolicy = CachePolicy()
    guardrails: GuardrailOverrides = GuardrailOverrides()
    limits: RequestLimits = RequestLimits()
    flags: KeyFlags = KeyFlags()

    @property
    def policy_id(self) -> str:
        return self.guardrails.policy_id

    def allows_model(self, model: str) -> bool:
        return any(fnmatchcase(model, pattern) for pattern in self.allowed_models)

    def allows_provider(self, provider: str) -> bool:
        # data-residency allow-list; None means every provider
        return self.allowed_providers is None or provider in self.allowed_providers

    def allows_deployment(self, deployment: Deployment) -> bool:
        if not self.allows_provider(deployment.provider):
            return False
        if not deployment.pricing.billed and not self.flags.allow_free_tier_providers:
            return False
        return deployment.tier != "premium" or self.flags.allow_premium
