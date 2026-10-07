from typing import Any

from pydantic import Field

from gg.core.schema import StrictModel


class MockFault(StrictModel):
    status: int = 500
    body: dict[str, Any] = Field(
        default_factory=lambda: {"error": {"message": "mock failure", "type": "server_error"}}
    )
    headers: dict[str, str] = Field(default_factory=dict)
    first_n_attempts: int | None = None
    after_chunks: int | None = None


class MockToolCall(StrictModel):
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)


class MockScenario(StrictModel):
    """deployment defaults.mock, optionally overridden per request with a top-level `mock` field (non-prod)"""

    text: str = "echo"
    ttft_ms: float = 0
    tokens_per_s: float = 0
    chunk_tokens: int = Field(default=4, ge=1)
    tool_call: MockToolCall | None = None
    fail: MockFault | None = None

    def reply(self, last_user: str | None) -> str:
        if self.text == "echo":
            return last_user or ""
        return self.text
