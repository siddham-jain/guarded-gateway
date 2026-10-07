from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, SecretStr

from gg.core.clock import Clock, SystemClock
from gg.core.schema import StrictModel
from gg.providers.catalog.capabilities import CapabilityChecker
from gg.providers.http import HttpClientFactory
from gg.providers.observer import NoopObserver, ProviderObserver
from gg.providers.state.base import ProviderStateStore
from gg.providers.state.memory import InMemoryStateStore
from gg.providers.usage import estimate_prompt_tokens


class AuthSpec(StrictModel):
    header: str = "Authorization"
    scheme: str | None = "Bearer"

    def headers(self, api_key: SecretStr | None) -> dict[str, str]:
        if api_key is None:
            return {}
        secret = api_key.get_secret_value()
        return {self.header: f"{self.scheme} {secret}" if self.scheme else secret}


class DataPolicy(StrictModel):
    jurisdiction: str | None = None
    processing_regions: tuple[str, ...] = ()
    trains_on_api_data: Literal["false", "opt_out", "true", "unknown"] = "unknown"
    retention_days: int | None = None
    zdr: Literal["none", "on_request", "self_serve", "default", "account_toggle", "unknown"] = "unknown"
    resale_allowed: Literal["yes", "no", "with_authorization", "unclear"] = "unclear"


class ProviderRuntime(StrictModel):
    """a provider resolved against settings: enabled, with its effective base url and credentials"""

    name: str
    type: str
    base_url: str
    api_key: SecretStr | None = None
    auth: AuthSpec = Field(default_factory=AuthSpec)
    quirks: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    max_in_flight: int | None = None
    data_policy: DataPolicy = Field(default_factory=DataPolicy)


@dataclass(slots=True)
class AdapterDeps:
    """what adapter factories receive from the composition root"""

    http: HttpClientFactory
    clock: Clock = field(default_factory=SystemClock)
    observer: ProviderObserver = field(default_factory=NoopObserver)
    checker: CapabilityChecker = field(default_factory=lambda: CapabilityChecker(estimate_prompt_tokens))
    state: ProviderStateStore = field(default_factory=InMemoryStateStore)
    env: str = "dev"
