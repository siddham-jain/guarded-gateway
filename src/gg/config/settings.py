from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseModel):
    max_body_bytes: int = 8 * 1024 * 1024
    body_read_timeout_s: float = 30
    sse_header_commit_s: float = 15
    sse_keepalive_s: float = 15
    drain_s: float = 20
    finalizer_timeout_s: float = 5
    cors_origins: tuple[str, ...] = ()


class DeadlineSettings(BaseModel):
    nonstream_s: float = 30
    stream_s: float = 120
    max_s: float = 300


class RuntimeSettings(BaseModel):
    cpu_workers: int = 2
    cpu_queue_max: int = 64
    background_task_max: int = 1000
    redis_max_connections: int = 64
    redis_pool_timeout_s: float = 0.5


class ProviderCredentials(BaseModel):
    api_key: SecretStr | None = None
    base_url: str | None = None


class MetricsSettings(BaseModel):
    enabled: bool = True
    bearer_token: SecretStr | None = None


class LangfuseSettings(BaseModel):
    """otlp trace export to langfuse; off unless both keys are set"""

    host: str = "http://localhost:3001"
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    sample_rate: Annotated[float, Field(ge=0, le=1)] = 1.0
    # scrubbed: placeholder-space prompt and reply text, masked and truncated; never the raw request
    capture_content: Literal["off", "scrubbed"] = "off"
    max_content_chars: Annotated[int, Field(ge=100, le=100_000)] = 4_000
    queue_max: Annotated[int, Field(ge=1)] = 2_048
    batch_max: Annotated[int, Field(ge=1)] = 64
    flush_interval_s: Annotated[float, Field(gt=0)] = 1.0
    timeout_s: Annotated[float, Field(gt=0)] = 5.0

    @property
    def enabled(self) -> bool:
        return self.public_key is not None and self.secret_key is not None and self.sample_rate > 0


class AuthFailureSettings(BaseModel):
    """per-ip limiter on failed auth; over the limit an ip gets 429 on /v1/* until the window ends"""

    enabled: bool = True
    max_failures: Annotated[int, Field(ge=1)] = 10
    window_s: Annotated[float, Field(gt=0)] = 60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GG_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    profile: Literal["local", "oracle", "cloudrun"] = "local"
    host: str = "127.0.0.1"
    port: int = 8000
    config_dir: Path = Path("config")
    keys_file: Path | None = None
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_format: Literal["json", "console"] = "json"
    redis_url: SecretStr | None = None
    jev_api_key: SecretStr | None = None
    promptguard_api_key: SecretStr | None = None
    # local guard model weights; unset keeps the ml guards off (they allow with reason ml_disabled)
    models_dir: Path | None = None
    model_profile: str = "prod"
    server: ServerSettings = ServerSettings()
    deadlines: DeadlineSettings = DeadlineSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    # keyed by provider name in models.yaml, e.g. GG_PROVIDERS__TOGETHER__API_KEY
    providers: dict[str, ProviderCredentials] = {}
    metrics: MetricsSettings = MetricsSettings()
    langfuse: LangfuseSettings = LangfuseSettings()
    auth_failures: AuthFailureSettings = AuthFailureSettings()
    # kill switch: /v1/* answers 503 except for keys tagged admin; health endpoints stay up
    maintenance: bool = False

    @model_validator(mode="after")
    def _check_prod(self) -> Self:
        if self.env == "prod" and self.log_format != "json":
            raise ValueError("prod requires GG_LOG_FORMAT=json")
        return self

    @property
    def resolved_keys_file(self) -> Path:
        return self.keys_file or self.config_dir / "keys.yaml"

    def provider(self, name: str) -> ProviderCredentials:
        return self.providers.get(name, ProviderCredentials())
