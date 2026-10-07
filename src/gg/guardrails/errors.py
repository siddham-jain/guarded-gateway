from gg.core.errors import GuardrailBlockedError, GuardrailUnavailableError, InvalidRequestError
from gg.guardrails.base import PolicyRef
from gg.guardrails.policy.schema import ResponseMessages


class GuardrailOverrideRejectedError(InvalidRequestError):
    default_code = "guardrail_override_rejected"

    def __init__(self, message: str) -> None:
        super().__init__(message, param="gg.guardrails")


def _headers(ref: PolicyRef, stage: str) -> dict[str, str]:
    return {"x-gg-policy": ref.header(), "x-gg-guardrails": "blocked", "x-gg-guardrail-stage": stage}


def blocked(
    request_id: str, ref: PolicyRef, messages: ResponseMessages, *, stage: str
) -> GuardrailBlockedError:
    # never names the guard, rule, score or matched text; operators correlate through the request id
    return GuardrailBlockedError(f"{messages.default} Reference: {request_id}", headers=_headers(ref, stage))


def unavailable(
    request_id: str, ref: PolicyRef, messages: ResponseMessages, *, stage: str
) -> GuardrailUnavailableError:
    return GuardrailUnavailableError(
        f"{messages.unavailable} Reference: {request_id}",
        retry_after_s=1,
        headers={**_headers(ref, stage), "x-should-retry": "true"},
    )
