"""runs each suite in process and returns the runner's result json; replay by default, live on request"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import orjson

from gg.cache.config import CacheConfig
from gg.cache.embedders import build_embedder
from gg.cache.eval.gate import baseline_from as cache_baseline
from gg.cache.eval.gate import pair_key, run_gate
from gg.cache.eval.pairs import load_pairs
from gg.cache.verifiers import build_verifier
from gg.config.yaml_loader import load_yaml_file
from gg.core.aio import CpuExecutor
from gg.core.clock import SystemClock
from gg.core.jsonutil import loads
from gg.core.schema import ChatRequest
from gg.evalgate.cassette import Cassette
from gg.evalgate.gates import SuiteSpec
from gg.guardrails.eval.items import load_items
from gg.guardrails.eval.runner import GuardrailEval, eval_key, report
from gg.guardrails.eval.runner import baseline_from as guardrails_baseline
from gg.guardrails.registry import RemoteClients
from gg.guardrails.setup import build_guardrails

POLICY_ID = "default"


@dataclass(frozen=True, slots=True)
class Live:
    """what a live run adds: local model weights and the hosted detector's key"""

    models_dir: Path
    promptguard_api_key: str | None = None
    jev_api_key: str | None = None


def _rebase(http: httpx2.AsyncClient, base_url: str) -> httpx2.AsyncClient:
    http.base_url = httpx2.URL(base_url)
    return http


def _probe_request() -> ChatRequest:
    return ChatRequest.model_validate({"model": "gg/auto", "messages": [{"role": "user", "content": "x"}]})


async def run_guardrails(
    spec: SuiteSpec, evals: Path, config: Path, baseline: dict[str, Any] | None, live: Live | None = None
) -> dict[str, Any]:
    """replay: rule packs plus jev's recorded answers, no weights. live: every guard, re-recording jev"""
    items = load_items(evals / spec.suite / "items")
    cassette = Cassette(evals / spec.suite / "cassettes" / "jev.json")
    clients: list[httpx2.AsyncClient] = []

    def client(name: str, base_url: str) -> httpx2.AsyncClient:
        transport = None
        if name == "jev":
            transport = cassette.replay() if live is None else cassette.record()
        clients.append(httpx2.AsyncClient(base_url=base_url, transport=transport))
        return clients[-1]

    keys: dict[str, str] = {"jev": "replay"} if len(cassette) else {}
    cpu = embedder = models_dir = None
    if live is not None:
        named = (("promptguard", live.promptguard_api_key), ("jev", live.jev_api_key))
        keys = {name: key for name, key in named if key}
        semantic = CacheConfig.model_validate(load_yaml_file(config / "cache.yaml")).semantic
        cpu = CpuExecutor(workers=2, queue_max=64)
        embedder = build_embedder(semantic.embedder, cpu, cache_dir=live.models_dir / "fastembed")
        models_dir = live.models_dir
    try:
        guardrails = build_guardrails(
            config / "policies",
            clock=SystemClock(),
            keys=[eval_key(POLICY_ID)],
            cpu=cpu,
            embedder=embedder,
            models_dir=models_dir,
            remote=RemoteClients(client=client, api_keys=keys) if keys else None,
        )
        await guardrails.start()
        runner = GuardrailEval(
            guardrails.policies, guardrails.engine, policy_id=POLICY_ID, probe=guardrails.tier2_probe
        )
        results = [await runner.run_item(item) for item in items]
        result = report(results, items, runner.policy(_probe_request()), baseline)
    finally:
        for c in clients:
            await c.aclose()
        if cpu is not None:
            cpu.shutdown()
    remote = ", ".join(f"{name} api" for name in keys) or "no remote detectors"
    if live is None:
        recorded = "jev replayed from the cassette" if keys else "no remote detectors"
        result["coverage"] = f"rule packs, {recorded}; model-backed guards off"
        return result
    if "jev" in keys:
        cassette.save()
    result["mode"] = "live"
    result["coverage"] = f"every guard in the policy: rules, local models, {remote}"
    return result


async def run_cache(
    spec: SuiteSpec, evals: Path, config: Path, baseline: dict[str, Any] | None, live: Live | None = None
) -> dict[str, Any]:
    """live: production embedder and jev verifier, recorded. replay: the recording at today's thresholds"""
    assert spec.embedder is not None
    assert spec.threshold is not None
    pairs = load_pairs(evals / spec.suite / "pairs.jsonl")
    cassette = Cassette(evals / spec.suite / "cassettes" / "jev.json")
    distances_path = evals / spec.suite / "cassettes" / "distances.json"
    recording = loads(distances_path.read_bytes()) if distances_path.is_file() else None
    cpu = CpuExecutor(workers=1, queue_max=8)
    try:
        if live is None and recording is None:
            # nothing recorded yet: the deterministic stand-in only catches tagging and filtering changes
            embedder = build_embedder(spec.embedder, cpu)
            result = await run_gate(pairs, embedder, tau=spec.threshold, baseline=baseline)
            result["coverage"] = f"{embedder.name} embedder at distance {spec.threshold:g}, num_sig filter"
            return result
        semantic = CacheConfig.model_validate(load_yaml_file(config / "cache.yaml")).semantic
        tau = semantic.distance_threshold
        transport = cassette.replay() if live is None else cassette.record()
        async with httpx2.AsyncClient(transport=transport) as http:
            key = "replay" if live is None else live.jev_api_key
            verifier = build_verifier(semantic.verifier, lambda base: _rebase(http, base), key)
            if semantic.verifier.type != "none" and verifier is None:
                raise ValueError("config/cache.yaml asks for the jev verifier; set GG_JEV_API_KEY")
            if live is None:
                assert recording is not None
                embedder = build_embedder(spec.embedder, cpu)
                result = await run_gate(
                    pairs,
                    embedder,
                    tau=tau,
                    baseline=baseline,
                    verifier=verifier,
                    recorded=recording["pairs"],
                )
                result["embedder"] = recording["embedder"]
            else:
                embedder = build_embedder(semantic.embedder, cpu, cache_dir=live.models_dir / "fastembed")
                result = await run_gate(pairs, embedder, tau=tau, baseline=baseline, verifier=verifier)
    finally:
        cpu.shutdown()
    if live is not None:
        result["mode"] = "live"
        cassette.save()
        by_id = {p.id: pair_key(p) for p in pairs}
        recorded = {by_id[i["id"]]: i["distance"] for i in result["items"]}
        payload = {"embedder": result["embedder"], "pairs": dict(sorted(recorded.items()))}
        distances_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")
    source = "live" if live is not None else "recorded"
    checked = f", {semantic.verifier.type} verifier at {semantic.verifier.min_score:g}" if verifier else ""
    name = result["embedder"]["name"]
    result["coverage"] = f"{source} {name} distances at {tau:g}, num_sig filter{checked}"
    return result


RUNNERS = {"guardrails": run_guardrails, "cache": run_cache}
BASELINES = {"guardrails": guardrails_baseline, "cache": cache_baseline}
