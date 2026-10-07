from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from gg.core.keypolicy import KeyPolicy
from gg.core.schema import ChatRequest, StrictModel
from gg.guardrails.base import Mode, OnError

type SemVer = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]

COMMON_FIELDS = frozenset(
    {"guard", "name", "mode", "on_error", "timeout_ms", "sample_rate", "when", "config"}
)


class Defaults(StrictModel):
    mode: Mode = Mode.ENFORCE
    on_error: OnError = OnError.FLAG
    timeout_ms: Annotated[int, Field(ge=1, le=5000)] = 150


class ResponseMessages(StrictModel):
    default: str = "The request was blocked by the gateway's content policy."
    unavailable: str = "A required safety check is temporarily unavailable. Retry later."
    output_blocked: str = "The response was withheld by the gateway's content policy."


class ResponseCfg(StrictModel):
    messages: ResponseMessages = ResponseMessages()


class WhenSpec(StrictModel):
    """guard applicability; every present clause must hold"""

    has_response_format: bool | None = None
    has_tools: bool | None = None
    stream: bool | None = None

    def matches(self, request: ChatRequest) -> bool:
        if self.has_response_format is not None:
            rf = request.response_format
            if (rf is not None and rf.type != "text") is not self.has_response_format:
                return False
        if self.has_tools is not None and request.has_tools() is not self.has_tools:
            return False
        return self.stream is None or request.stream is self.stream


class GuardEntry(StrictModel):
    """one guard in a policy; yaml is flat, everything but the common fields is the guard's own config"""

    guard: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")]
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")] | None = None
    mode: Mode | None = None
    on_error: OnError | None = None
    timeout_ms: Annotated[int | None, Field(ge=1, le=5000)] = None
    sample_rate: Annotated[float, Field(ge=0, le=1)] = 1.0
    when: WhenSpec | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_config(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "config" in data:
            return data
        items: dict[str, Any] = {str(k): v for k, v in data.items()}  # pyright: ignore[reportUnknownVariableType]
        config = {k: v for k, v in items.items() if k not in COMMON_FIELDS}
        return {**{k: v for k, v in items.items() if k in COMMON_FIELDS}, "config": config}

    @property
    def key(self) -> str:
        return self.name or self.guard


def _unique_names(guards: tuple[GuardEntry, ...], stage: str) -> None:
    seen: set[str] = set()
    for entry in guards:
        if entry.key in seen:
            raise ValueError(f"duplicate {stage} guard name '{entry.key}'; set `name:` to tell them apart")
        seen.add(entry.key)


class PhaseDeadlines(StrictModel):
    pre: Annotated[int, Field(ge=1, le=5000)] = 150
    # tier 2-3, run by the guard probe alongside the router and the semantic cache
    parallel: Annotated[int, Field(ge=1, le=10_000)] = 300


class InputSection(StrictModel):
    placeholder_hint: bool = True
    # set true to let secrets/pii guards run in shadow or off, sending raw values upstream
    allow_unredacted_upstream: bool = False
    phase_deadline_ms: PhaseDeadlines = PhaseDeadlines()
    guards: tuple[GuardEntry, ...] = ()

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        _unique_names(self.guards, "input")
        return self


class StreamingCfg(StrictModel):
    mode: Literal["windowed", "buffer"] = "windowed"
    first_window_chars: Annotated[int, Field(ge=1)] = 40
    window_chars: Annotated[int, Field(ge=16)] = 200
    overlap_chars: Annotated[int, Field(ge=0)] = 50
    holdback_max_chars: Annotated[int, Field(ge=0, le=512)] = 64
    abort: Literal["graceful", "error_frame"] = "graceful"
    tool_args_guards: tuple[str, ...] = ("secrets_out", "pii_leak")

    @model_validator(mode="after")
    def _check_sizes(self) -> Self:
        if self.overlap_chars >= self.window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        return self


class OutputSection(StrictModel):
    streaming: StreamingCfg = StreamingCfg()
    guards: tuple[GuardEntry, ...] = ()

    @model_validator(mode="after")
    def _check_names(self) -> Self:
        _unique_names(self.guards, "output")
        return self


class OverrideMatch(StrictModel):
    key_tags_any: tuple[str, ...] = ()
    key_ids: tuple[str, ...] = ()
    models: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not (self.key_tags_any or self.key_ids or self.models):
            raise ValueError("an override match needs at least one of key_tags_any, key_ids, models")
        return self

    def matches(self, key: KeyPolicy, model: str) -> bool:
        if self.key_tags_any and not key.tags.intersection(self.key_tags_any):
            return False
        if self.key_ids and key.id not in self.key_ids:
            return False
        return not self.models or any(fnmatchcase(model, pattern) for pattern in self.models)


class Override(StrictModel):
    name: str
    match: OverrideMatch
    # dotted paths: input.guards.<name>.<field>, output.streaming.<field>, input.<setting>
    patch: dict[str, Any] = Field(min_length=1)


class TighteningCfg(StrictModel):
    allow_request: bool = True


class PolicyDoc(StrictModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")]
    version: SemVer
    description: str = ""
    defaults: Defaults = Defaults()
    response: ResponseCfg = ResponseCfg()
    input: InputSection = InputSection()
    output: OutputSection = OutputSection()
    overrides: tuple[Override, ...] = ()
    tightening: TighteningCfg = TighteningCfg()

    @model_validator(mode="after")
    def _unique_overrides(self) -> Self:
        names = [o.name for o in self.overrides]
        if len(names) != len(set(names)):
            raise ValueError("override names must be unique")
        return self
