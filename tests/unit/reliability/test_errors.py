import pytest

from gg.core.errors import (
    ContentFilterError,
    ContextLengthError,
    GGError,
    InvalidRequestError,
    RateLimitedError,
    ServiceUnavailableError,
    UpstreamError,
    UpstreamTimeoutError,
)
from gg.reliability.errors import Failure, client_error, error_from_attempts
from tests.unit.reliability.fakes import deployment, err

A = deployment("a/m")
B = deployment("b/m")


def failed(**kwargs: object) -> Failure:
    return Failure(A, error=err(**kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("failures", "deadline_hit", "error_type", "code", "retry_after", "should_retry"),
    [
        (
            [failed(kind="fallback", status=400, code="context_length"), Failure(B, skip="context_window")],
            False,
            ContextLengthError,
            "context_length_exceeded",
            None,
            "false",
        ),
        (
            [Failure(A, skip="circuit_open", retry_in_s=4.2)],
            False,
            ServiceUnavailableError,
            "no_healthy_deployment",
            4.2,
            "true",
        ),
        (
            [Failure(A, skip="circuit_open", retry_in_s=300)],
            False,
            ServiceUnavailableError,
            "no_healthy_deployment",
            300,
            "false",
        ),
        (
            [Failure(A, skip="no_adapter")],
            False,
            ServiceUnavailableError,
            "no_healthy_deployment",
            None,
            "false",
        ),
        ([], False, ServiceUnavailableError, "no_healthy_deployment", None, "false"),
        (
            [
                failed(kind="quota_minute", status=429, retry_after_s=9),
                failed(kind="quota_minute", status=429),
            ],
            False,
            RateLimitedError,
            "upstream_rate_limited",
            9,
            "true",
        ),
        (
            [failed(kind="retryable", status=504, code="ttft_timeout")],
            False,
            UpstreamTimeoutError,
            "upstream_timeout",
            None,
            "false",
        ),
        (
            [failed(kind="retryable", status=500)],
            True,
            UpstreamTimeoutError,
            "upstream_timeout",
            None,
            "false",
        ),
        ([], True, UpstreamTimeoutError, "upstream_timeout", None, "false"),
        (
            [failed(kind="auth", status=401), failed(kind="quota_day", status=429)],
            False,
            UpstreamError,
            "upstream_account_error",
            None,
            "false",
        ),
        (
            [failed(kind="fallback", status=529, code="overloaded"), failed(kind="retryable", status=500)],
            False,
            ServiceUnavailableError,
            "upstream_overloaded",
            2,
            "true",
        ),
        (
            [failed(kind="retryable", status=500), Failure(B, skip="circuit_open", retry_in_s=10)],
            False,
            ServiceUnavailableError,
            "upstream_overloaded",
            2,
            "true",
        ),
        ([failed(kind="retryable", status=500)], False, UpstreamError, "upstream_error", None, "false"),
    ],
)
def test_exhaustion_rules(
    failures: list[Failure],
    deadline_hit: bool,
    error_type: type[GGError],
    code: str,
    retry_after: float | None,
    should_retry: str,
) -> None:
    error = error_from_attempts(failures, deadline_hit=deadline_hit)
    assert type(error) is error_type
    assert error.code == code
    assert error.retry_after_s == retry_after
    assert error.response_headers()["x-should-retry"] == should_retry


def test_upstream_error_lists_attempts_without_messages() -> None:
    error = error_from_attempts([failed(kind="retryable", status=500, code="server_error")])
    assert error.to_body()["error"]["gg"] == {
        "attempts": [{"provider": "fake", "status": 500, "code": "server_error"}]
    }


def test_client_error_mapping() -> None:
    assert isinstance(client_error(err("content_filter", status=400)), ContentFilterError)
    error = client_error(err("client", status=400, code="invalid_value"))
    assert type(error) is InvalidRequestError
    assert error.message == "fake: scripted failure"
