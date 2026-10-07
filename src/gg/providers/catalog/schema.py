from datetime import date
from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from gg.core.deployment import (
    Capabilities,
    ImageUrlMode,
    LongContextPricing,
    Pricing,
    SamplingParams,
    ThinkingMode,
)
from gg.core.schema import ReasoningEffort, StrictModel
from gg.providers.runtime import AuthSpec, DataPolicy

type ProviderType = Literal["openai_compat", "anthropic", "gemini", "mock"]
type Billing = Literal["paid", "free_tier", "subscription"]
type Status = Literal["active", "preview", "deprecated", "disabled"]


class CapabilitiesSpec(StrictModel):
    tools: bool = True
    forced_tool_choice: bool = True
    parallel_tool_calls: bool = True
    prefill: bool = True
    sampling_params: SamplingParams = "full"
    effort_levels: tuple[ReasoningEffort, ...] = ()
    effort_clamp: dict[ReasoningEffort, ReasoningEffort] = Field(default_factory=dict)
    tools_require_effort: ReasoningEffort | None = None
    json_schema: bool = True
    json_schema_strict: bool = True
    json_object: bool = True
    vision: bool = False
    image_url: ImageUrlMode = "provider_fetch"
    pdf: bool = False
    audio: bool = False
    n: bool = False
    logprobs: bool = False
    stop_max: int | None = None
    temperature_max: float = 2.0
    thinking_mode: ThinkingMode = "none"
    mid_conversation_system: bool = True
    max_output: int = Field(default=8192, ge=1)
    context: int = Field(default=128_000, ge=1)
    min_max_tokens: int = Field(default=1, ge=1)
    cache_min_tokens: int | None = None
    honors_cancellation: bool = True

    def build(self) -> Capabilities:
        data = self.model_dump()
        data["effort_levels"] = frozenset(self.effort_levels)
        data["effort_clamp"] = dict(self.effort_clamp)
        return Capabilities(**data)


class LongContextSpec(StrictModel):
    threshold_input_tokens: int
    input_mult: Decimal
    output_mult: Decimal


class PricingSpec(StrictModel):
    """usd per 1m tokens from effective_from (utc date, inclusive)"""

    effective_from: date
    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)
    cached_input: Decimal | None = Field(default=None, ge=0)
    cache_write: Decimal | None = Field(default=None, ge=0)
    cache_write_1h: Decimal | None = Field(default=None, ge=0)
    long_context: LongContextSpec | None = None

    def build(self) -> Pricing:
        lc = self.long_context
        return Pricing(
            effective_from=self.effective_from,
            input=self.input,
            output=self.output,
            cached_input=self.cached_input,
            cache_write=self.cache_write,
            cache_write_1h=self.cache_write_1h,
            long_context=LongContextPricing(lc.threshold_input_tokens, lc.input_mult, lc.output_mult)
            if lc
            else None,
        )


class ProviderSpec(StrictModel):
    type: ProviderType
    base_url: str | None = None
    base_path: str = ""
    requires_key: bool = True
    auth: AuthSpec = Field(default_factory=AuthSpec)
    quirks: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    max_in_flight: int | None = Field(default=None, ge=1)
    billing: Billing = "paid"
    data_policy: DataPolicy = Field(default_factory=DataPolicy)
    enabled: bool = True


class DeploymentSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+/.+$")
    provider: str | None = None
    upstream_model: str | None = None
    capabilities: CapabilitiesSpec = Field(default_factory=CapabilitiesSpec)
    defaults: dict[str, Any] = Field(default_factory=dict)
    pricing: list[PricingSpec] = Field(default_factory=list)
    status: Status = "active"
    shutdown_date: date | None = None
    billing: Billing | None = None
    reliability: dict[str, Any] = Field(default_factory=dict)
    quirks: dict[str, Any] = Field(default_factory=dict)
    tier: str | None = None
    tags: tuple[str, ...] = ()
    data_policy: DataPolicy | None = None
    public: bool = True

    @property
    def provider_name(self) -> str:
        return self.provider or self.id.split("/", 1)[0]

    @property
    def upstream(self) -> str:
        return self.upstream_model or self.id.split("/", 1)[1]


class HostSpec(StrictModel):
    """one host serving a canonical model; becomes deployment `<provider>/<canonical id>` unless id is set"""

    provider: str
    upstream_model: str
    id: str | None = None
    quantization: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)
    pricing: list[PricingSpec] = Field(default_factory=list)
    status: Status = "active"
    billing: Billing | None = None
    reliability: dict[str, Any] = Field(default_factory=dict)
    quirks: dict[str, Any] = Field(default_factory=dict)
    data_policy: DataPolicy | None = None


class CanonicalModelSpec(StrictModel):
    hf_repo: str | None = None
    native_precision: str | None = None
    capabilities: CapabilitiesSpec = Field(default_factory=CapabilitiesSpec)
    defaults: dict[str, Any] = Field(default_factory=dict)
    selection: Literal["order", "price"] = "order"
    allow_quantizations: tuple[str, ...] | None = None
    public: bool = True
    hosts: list[HostSpec] = Field(min_length=1)


class GroupSpec(StrictModel):
    chain: list[str] = Field(min_length=1)
    chain_with_tools: list[str] | None = None


class AliasSpec(StrictModel):
    router: bool = False
    group: str | None = None
    groups: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.router and not self.groups:
            raise ValueError("router aliases need groups")
        if not self.router and self.group is None:
            raise ValueError("aliases need a group (or router: true with groups)")
        return self


class ProfileSpec(StrictModel):
    groups: dict[str, GroupSpec] = Field(default_factory=dict)
    aliases: dict[str, AliasSpec] = Field(default_factory=dict)
    only_providers: list[str] | None = None
    disable_providers: list[str] = Field(default_factory=list)


class ModelsConfig(StrictModel):
    """config/models.yaml; the reliability block is c4's and kept opaque here"""

    version: Literal[1] = 1
    quirk_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    capability_profiles: dict[str, CapabilitiesSpec] = Field(default_factory=dict)
    providers: dict[str, ProviderSpec]
    reliability: dict[str, Any] = Field(default_factory=dict)
    deployments: list[DeploymentSpec] = Field(default_factory=list)
    models: dict[str, CanonicalModelSpec] = Field(default_factory=dict)
    groups: dict[str, GroupSpec] = Field(default_factory=dict)
    aliases: dict[str, AliasSpec] = Field(default_factory=dict)
    bare_names: bool = True
    profiles: dict[str, ProfileSpec] = Field(default_factory=dict)
