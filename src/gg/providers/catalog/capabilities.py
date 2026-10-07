from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from gg.core.deployment import EFFORT_ORDER, Deployment
from gg.core.schema import ChatRequest, FilePart, ImagePart, InputAudioPart

_SAMPLING = ("temperature", "top_p")
_REASONING_STRIPPED = ("temperature", "top_p", "logprobs", "top_logprobs")


@dataclass(frozen=True, slots=True)
class AdjustPolicy:
    tool_choice_unsupported: Literal["downgrade", "reject"] = "reject"


@dataclass(frozen=True, slots=True)
class CheckResult:
    rejects: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()
    ignored_params: tuple[str, ...] = ()
    patch: dict[str, Any] = field(default_factory=lambda: {})

    @property
    def ok(self) -> bool:
        return not self.rejects


def _nearest_effort(value: str, levels: frozenset[str]) -> str | None:
    if not levels or value not in EFFORT_ORDER:
        return None
    target = EFFORT_ORDER.index(value)
    ranked = sorted(
        levels & set(EFFORT_ORDER),
        key=lambda lv: (abs(EFFORT_ORDER.index(lv) - target), -EFFORT_ORDER.index(lv)),
    )
    return ranked[0] if ranked else None


def _forced(choice: Any) -> bool:
    return choice == "required" or (choice is not None and not isinstance(choice, str))


class CapabilityChecker:
    """pure pre-send check: rejects what a deployment cannot serve, adjusts what it can serve differently"""

    def __init__(self, estimate_prompt_tokens: Any = None) -> None:
        self._estimate = estimate_prompt_tokens

    def check(self, req: ChatRequest, dep: Deployment, policy: AdjustPolicy | None = None) -> CheckResult:
        policy = policy or AdjustPolicy()
        caps = dep.capabilities
        rejects: list[str] = []
        adjustments: list[str] = []
        ignored: list[str] = []
        patch: dict[str, Any] = {}

        if req.tools and not caps.tools:
            rejects.append("tools")
        if _forced(req.tool_choice) and not caps.forced_tool_choice:
            if policy.tool_choice_unsupported == "downgrade":
                patch["tool_choice"] = "auto"
                adjustments.append("tool_choice:forced->auto")
            else:
                rejects.append("forced_tool_choice")
        if req.parallel_tool_calls is not None and not caps.parallel_tool_calls:
            patch["parallel_tool_calls"] = None
            ignored.append("parallel_tool_calls")
        if req.response_format is not None:
            kind = req.response_format.type
            if kind == "json_schema" and not caps.json_schema:
                rejects.append("json_schema")
            if kind == "json_object" and not caps.json_object:
                rejects.append("json_object")
        rejects.extend(self._content_rejects(req, dep))
        if req.n > 1 and not caps.n:
            rejects.append("n")
        if (req.logprobs or req.top_logprobs) and not caps.logprobs:
            rejects.append("logprobs")
        if req.stop and caps.stop_max is not None and len(req.stop) > caps.stop_max:
            rejects.append("stop")
        if req.messages[-1].role == "assistant" and not caps.prefill:
            rejects.append("prefill")

        effort = self._effort(req, dep, patch, adjustments, ignored)
        self._sampling(req, dep, effort, patch, adjustments, ignored)
        self._max_tokens(req, dep, patch, adjustments)
        if self._estimate is not None and self._estimate(req) > caps.context:
            rejects.append("context_length")
        return CheckResult(tuple(rejects), tuple(adjustments), tuple(ignored), patch)

    def apply(self, req: ChatRequest, result: CheckResult) -> ChatRequest:
        if not result.patch:
            return req
        return req.model_copy(update=result.patch)

    def compatible(
        self, req: ChatRequest, deployments: Sequence[Deployment], policy: AdjustPolicy | None = None
    ) -> list[Deployment]:
        return [d for d in deployments if self.check(req, d, policy).ok]

    def _content_rejects(self, req: ChatRequest, dep: Deployment) -> list[str]:
        caps = dep.capabilities
        found: set[str] = set()
        for message in req.messages:
            if isinstance(message.content, str) or message.content is None:
                continue
            for part in message.content:
                if isinstance(part, ImagePart):
                    if not caps.vision:
                        found.add("vision")
                    elif not part.image_url.url.startswith("data:") and caps.image_url != "provider_fetch":
                        found.add("image_url")
                elif isinstance(part, InputAudioPart) and not caps.audio:
                    found.add("audio")
                elif isinstance(part, FilePart) and not caps.pdf:
                    found.add("pdf")
        return sorted(found)

    def _effort(
        self,
        req: ChatRequest,
        dep: Deployment,
        patch: dict[str, Any],
        adjustments: list[str],
        ignored: list[str],
    ) -> str | None:
        caps = dep.capabilities
        requested = req.reasoning_effort
        default = dep.defaults.get("reasoning_effort")
        if not caps.effort_levels:
            if requested is not None:
                patch["reasoning_effort"] = None
                ignored.append("reasoning_effort")
            return None
        effort = requested or (default if isinstance(default, str) else None)
        if effort is not None and effort not in caps.effort_levels:
            target = caps.effort_clamp.get(effort) or _nearest_effort(effort, caps.effort_levels)
            if target is not None:
                adjustments.append(f"reasoning_effort:{effort}->{target}")
                effort = target
        if req.tools and caps.tools_require_effort is not None and effort != caps.tools_require_effort:
            adjustments.append(f"reasoning_effort:{effort}->{caps.tools_require_effort}")
            effort = caps.tools_require_effort
        if effort is not None and effort != requested:
            patch["reasoning_effort"] = effort
        return effort

    def _sampling(
        self,
        req: ChatRequest,
        dep: Deployment,
        effort: str | None,
        patch: dict[str, Any],
        adjustments: list[str],
        ignored: list[str],
    ) -> None:
        caps = dep.capabilities
        mode = caps.sampling_params
        strip: tuple[str, ...] = ()
        if mode == "none":
            strip = _SAMPLING
        elif mode == "when_effort_none" and effort not in (None, "none"):
            strip = _REASONING_STRIPPED
        elif mode == "temp_or_top_p" and req.temperature is not None and req.top_p is not None:
            strip = ("top_p",)
        elif mode == "advisory" and req.temperature is not None and req.temperature < 1.0:
            strip = ("temperature",)
        for name in strip:
            if getattr(req, name) is not None:
                patch[name] = None
                ignored.append(name)
        temperature = req.temperature
        if "temperature" not in patch and temperature is not None and temperature > caps.temperature_max:
            patch["temperature"] = caps.temperature_max
            adjustments.append(f"temperature:{temperature}->{caps.temperature_max}")

    def _max_tokens(
        self, req: ChatRequest, dep: Deployment, patch: dict[str, Any], adjustments: list[str]
    ) -> None:
        caps = dep.capabilities
        requested = req.max_completion_tokens
        default = dep.defaults.get("max_completion_tokens")
        value = (
            requested if requested is not None else (default if isinstance(default, int) else caps.max_output)
        )
        clamped = max(caps.min_max_tokens, min(value, caps.max_output))
        if requested is not None and clamped != requested:
            adjustments.append(f"max_completion_tokens:{requested}->{clamped}")
        if clamped != requested:
            patch["max_completion_tokens"] = clamped
