"""runs each suite in process with no network and returns the runner's result json"""

from pathlib import Path
from typing import Any

from gg.cache.embedders import build_embedder
from gg.cache.eval.gate import baseline_from as cache_baseline
from gg.cache.eval.gate import run_gate
from gg.cache.eval.pairs import load_pairs
from gg.core.aio import CpuExecutor
from gg.core.clock import SystemClock
from gg.core.schema import ChatRequest
from gg.evalgate.gates import SuiteSpec
from gg.guardrails.eval.items import load_items
from gg.guardrails.eval.runner import GuardrailEval, eval_key, report
from gg.guardrails.eval.runner import baseline_from as guardrails_baseline
from gg.guardrails.setup import build_guardrails

POLICY_ID = "default"


async def run_guardrails(
    spec: SuiteSpec, evals: Path, config: Path, baseline: dict[str, Any] | None
) -> dict[str, Any]:
    # no models_dir and no remote clients: model-backed guards allow everything, so only rule packs decide
    guardrails = build_guardrails(config / "policies", clock=SystemClock(), keys=[eval_key(POLICY_ID)])
    items = load_items(evals / spec.suite / "items")
    runner = GuardrailEval(guardrails.policies, guardrails.engine, policy_id=POLICY_ID)
    results = [await runner.run_item(item) for item in items]
    probe = ChatRequest.model_validate({"model": "gg/auto", "messages": [{"role": "user", "content": "x"}]})
    result = report(results, items, runner.policy(probe), baseline)
    result["coverage"] = "rule-based guards only; model-backed and remote guards off"
    return result


async def run_cache(
    spec: SuiteSpec, evals: Path, config: Path, baseline: dict[str, Any] | None
) -> dict[str, Any]:
    assert spec.embedder is not None
    assert spec.threshold is not None
    cpu = CpuExecutor(workers=1, queue_max=8)
    try:
        embedder = build_embedder(spec.embedder, cpu)
        pairs = load_pairs(evals / spec.suite / "pairs.jsonl")
        result = await run_gate(pairs, embedder, tau=spec.threshold, baseline=baseline)
    finally:
        cpu.shutdown()
    result["coverage"] = f"{embedder.name} embedder at distance {spec.threshold:g}, num_sig tag filter"
    return result


RUNNERS = {"guardrails": run_guardrails, "cache": run_cache}
BASELINES = {"guardrails": guardrails_baseline, "cache": cache_baseline}
