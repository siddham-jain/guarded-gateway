"""suite gates (C11 §3.5): item regressions minus accepted changes, then metric floors, ceilings and drops"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from gg.cache.config import EmbedderConfig
from gg.config.yaml_loader import load_yaml_text
from gg.core.schema import StrictModel

ITEM_REGRESSIONS = "item_regressions"


class GateSpec(StrictModel):
    mode: Literal["gate", "report"] = "gate"
    min: float | None = None
    # floor for replay runs, which have no model weights; `min` then applies to --live only
    replay_min: float | None = None
    max: float | None = None
    max_drop: float | None = None


class SuiteSpec(StrictModel):
    suite: Literal["guardrails", "cache"]
    version: str
    gates: dict[str, GateSpec]
    # cache only: the embedder and distance threshold the replay gate scores pairs with
    embedder: EmbedderConfig | None = None
    threshold: Annotated[float, Field(ge=0, le=2)] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        for name, gate in self.gates.items():
            bounds = (gate.min, gate.max, gate.max_drop)
            if name != ITEM_REGRESSIONS and all(b is None for b in bounds):
                raise ValueError(f"gate {name} needs min, max or max_drop")
        if self.suite == "cache" and (self.embedder is None or self.threshold is None):
            raise ValueError("the cache suite needs embedder and threshold")
        return self


class AcceptedChange(StrictModel):
    item: str
    from_: Annotated[str, Field(alias="from")]
    to: str
    reason: Annotated[str, Field(min_length=1)]
    pr: int | None = None


_ACCEPTED = TypeAdapter(list[AcceptedChange])


def parse_suite(text: str) -> SuiteSpec:
    return SuiteSpec.model_validate(load_yaml_text(text))


def parse_accepted(text: str | None) -> list[AcceptedChange]:
    if text is None:
        return []
    return _ACCEPTED.validate_python(load_yaml_text(text) or [])


def added_entries(head: Sequence[AcceptedChange], base: Sequence[AcceptedChange]) -> list[AcceptedChange]:
    """entries present now but not at the base ref, so an old exception can't excuse a new regression"""
    return [entry for entry in head if entry not in base]


def is_accepted(regression: dict[str, str], accepted: Sequence[AcceptedChange]) -> bool:
    return any(a.item == regression["id"] and a.to == regression["to"] for a in accepted)


def _item_gate(
    spec: GateSpec, blocking: list[dict[str, str]], accepted: list[dict[str, str]]
) -> dict[str, Any]:
    detail = ", ".join(f"{r['id']} ({r['from']}->{r['to']})" for r in blocking) or "none"
    if accepted:
        detail += f"; accepted: {', '.join(r['id'] for r in accepted)}"
    return _gate(ITEM_REGRESSIONS, spec, ok=not blocking, detail=detail)


def _metric_gate(
    name: str, spec: GateSpec, metrics: dict[str, Any], baseline: dict[str, Any] | None, *, live: bool
) -> dict[str, Any]:
    value = metrics.get(name, {}).get("value")
    if value is None:
        return _gate(name, spec, ok=False, detail="metric not reported")
    problems: list[str] = []
    checks: list[str] = []
    floor = spec.min if live or spec.replay_min is None else spec.replay_min
    if floor is not None:
        ok = value >= floor
        (checks if ok else problems).append(f"{value:g} {'>=' if ok else '<'} {floor:g}")
    if spec.max is not None:
        ok = value <= spec.max
        (checks if ok else problems).append(f"{value:g} {'<=' if ok else '>'} {spec.max:g}")
    if spec.max_drop is not None:
        before = (baseline or {}).get("metrics", {}).get(name)
        if before is None:
            checks.append("no baseline value")
        else:
            drop = round(before - value, 6)
            (checks if drop <= spec.max_drop else problems).append(
                f"drop {drop:g} from baseline {before:g} (max {spec.max_drop:g})"
            )
    return _gate(name, spec, ok=not problems, detail="; ".join(problems or checks))


def _gate(name: str, spec: GateSpec, *, ok: bool, detail: str) -> dict[str, Any]:
    failed = "fail" if spec.mode == "gate" else "warn"
    return {"name": name, "mode": spec.mode, "status": "pass" if ok else failed, "detail": detail}


def evaluate(
    result: dict[str, Any],
    spec: SuiteSpec,
    baseline: dict[str, Any] | None,
    accepted: Sequence[AcceptedChange],
) -> dict[str, Any]:
    """replaces the runner's gates and status with the suite.yaml gates"""
    raw: list[dict[str, str]] = result["regressions"]
    blocking = [r for r in raw if not is_accepted(r, accepted)]
    excused = [r for r in raw if is_accepted(r, accepted)]
    gates = [
        _item_gate(gate, blocking, excused)
        if name == ITEM_REGRESSIONS
        else _metric_gate(name, gate, result["metrics"], baseline, live=result.get("mode") == "live")
        for name, gate in spec.gates.items()
    ]
    return {
        **result,
        "regressions": blocking,
        "accepted_regressions": excused,
        "gates": gates,
        "status": "fail" if any(g["status"] == "fail" for g in gates) else "pass",
    }
