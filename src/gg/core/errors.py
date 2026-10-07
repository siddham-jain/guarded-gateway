from collections.abc import Mapping
from datetime import datetime
from typing import Any, ClassVar, Literal

from gg.core.jsonutil import JSONValue


class GGError(Exception):
    """client-facing error; rendered in the OpenAI error shape by the api layer"""

    status: ClassVar[int] = 500
    type: ClassVar[str] = "api_error"
    default_code: ClassVar[str] = "internal_error"
    client_should_retry: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str | None = None,
        retry_after_s: float | None = None,
        headers: Mapping[str, str] | None = None,
        details: Mapping[str, JSONValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code or self.default_code
        self.retry_after_s = retry_after_s
        self.headers: dict[str, str] = dict(headers or {})
        self.details: dict[str, JSONValue] = dict(details or {})

    def to_body(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }
        if self.details:
            error["gg"] = self.details
        return {"error": error}

    def response_headers(self) -> dict[str, str]:
        headers = {"x-should-retry": "true" if self.client_should_retry else "false"}
        if self.retry_after_s is not None:
            headers["retry-after"] = str(max(1, round(self.retry_after_s)))
        headers.update(self.headers)
        return headers


class InvalidRequestError(GGError):
    status = 400
    type = "invalid_request_error"
    default_code = "invalid_value"


class GuardrailBlockedError(InvalidRequestError):
    default_code = "guardrail_blocked"


class ContentFilterError(InvalidRequestError):
    default_code = "content_filter"


class ContextLengthError(InvalidRequestError):
    default_code = "context_length_exceeded"


class AuthenticationError(GGError):
    status = 401
    type = "invalid_request_error"
    default_code = "invalid_api_key"


class QuotaExceededError(GGError):
    status = 402
    type = "insufficient_quota"
    default_code = "budget_exceeded"


class PermissionDeniedError(GGError):
    status = 403
    type = "invalid_request_error"
    default_code = "model_not_allowed"


class NotFoundError(GGError):
    status = 404
    type = "invalid_request_error"
    default_code = "model_not_found"


class RequestTimeoutError(GGError):
    status = 408
    type = "invalid_request_error"
    default_code = "request_timeout"
    client_should_retry = False


class PayloadTooLargeError(GGError):
    status = 413
    type = "invalid_request_error"
    default_code = "request_too_large"


class UnsupportedMediaTypeError(GGError):
    status = 415
    type = "invalid_request_error"
    default_code = "unsupported_media_type"


class RateLimitedError(GGError):
    status = 429
    type = "rate_limit_error"
    default_code = "rate_limit_exceeded"
    client_should_retry = True


class InternalError(GGError):
    status = 500
    type = "api_error"
    default_code = "internal_error"


class UpstreamError(GGError):
    status = 502
    type = "api_error"
    default_code = "upstream_error"


class ServiceUnavailableError(GGError):
    status = 503
    type = "service_unavailable_error"
    default_code = "no_healthy_deployment"


class GuardrailUnavailableError(ServiceUnavailableError):
    default_code = "guardrail_unavailable"


class UpstreamTimeoutError(GGError):
    status = 504
    type = "api_error"
    default_code = "upstream_timeout"


type ProviderErrorKind = Literal[
    "retryable", "fallback", "client", "auth", "quota_minute", "quota_day", "content_filter"
]


class ProviderError(Exception):
    """raised by adapters, consumed by the executor; never rendered to clients directly"""

    def __init__(
        self,
        kind: ProviderErrorKind,
        *,
        provider: str,
        status: int,
        code: str | None = None,
        retry_after_s: float | None = None,
        message: str = "",
        upstream_request_id: str | None = None,
        quota_reset_at: datetime | None = None,
        scope: Literal["deployment", "provider"] = "deployment",
        committed: bool = False,
        deployment_id: str | None = None,
        violations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"{provider} {status} {kind}: {message}")
        self.kind: ProviderErrorKind = kind
        self.provider = provider
        self.status = status
        self.code = code
        self.retry_after_s = retry_after_s
        self.message = message
        self.upstream_request_id = upstream_request_id
        self.quota_reset_at = quota_reset_at
        self.scope: Literal["deployment", "provider"] = scope
        self.committed = committed
        self.deployment_id = deployment_id
        self.violations = violations
