"""guard model load time, resident memory and per-guard latency on a fixed corpus.

opt-in and needs the weights in .models/: GG_ML_BENCH=1 pytest tests/unit/guardrails/ml/test_bench_ml.py -s
memory: a fresh interpreter per model (after importing the ml libraries), then all models together.
"""

import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from gg.core.aio import CpuExecutor
from gg.guardrails.base import GuardContext, GuardFinding, Segment
from gg.guardrails.eval.stats import percentile
from gg.guardrails.ml.artefacts import MODELS
from gg.guardrails.ml.grounding import Grounding, GroundingCfg
from gg.guardrails.ml.pii_ner import PiiNer, PiiNerCfg
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.ml.topic import Topic, TopicCfg, build_index
from gg.guardrails.ml.toxicity import Toxicity, ToxicityCfg
from gg.guardrails.vault import GuardVault
from tests.conftest import make_request
from tests.unit.guardrails.ml.support import FASTEMBED_DIR, store
from tests.unit.guardrails.support import DEFAULT_POLICY, ROOT
from tests.unit.guardrails.test_bench import corpus

pytestmark = pytest.mark.skipif(os.environ.get("GG_ML_BENCH") != "1", reason="set GG_ML_BENCH=1 to run")

REPEAT = 5
WARMUP = 1
TOXICITY = "granite-guardian-hap-38m"
HHEM = "hhem-2.1-open"


def rss_mb() -> float:
    """linux: resident set; macos: phys_footprint (resident drops when the os compresses idle pages)"""
    gc.collect()
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    # rusage_info_v0: 16-byte uuid, six u64 counters, resident_size, phys_footprint
    info = (ctypes.c_uint64 * 16)()
    if ctypes.CDLL("libc.dylib").proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return info[9] / 2**20


def load(key: str) -> Any:
    if key == "presidio":
        from gg.guardrails.ml.presidio import PresidioAnalyzer

        return PresidioAnalyzer("en_core_web_sm")
    if key == HHEM:
        from gg.guardrails.ml.hhem import HhemScorer

        return HhemScorer(MODELS[HHEM], store().fetch(MODELS[HHEM]))
    if key == "bge-small":
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        list(model.embed(["warm up"]))
        return model
    from gg.guardrails.ml.onnx import OnnxTextClassifier

    return OnnxTextClassifier.load(MODELS[key], store().fetch(MODELS[key]))


def available() -> list[str]:
    keys = [m for m in (TOXICITY, HHEM) if store().present(MODELS[m])]
    keys.append("presidio")
    if any(FASTEMBED_DIR.glob("models--Qdrant--bge-small-en-v1.5*")):
        keys.append("bge-small")
    return keys


def offline_fastembed() -> None:
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def measure(keys: Sequence[str]) -> None:
    """child process: one json line per model with load seconds and memory added"""
    import onnxruntime  # noqa: F401 - the baseline includes the runtimes every guard shares
    import presidio_analyzer  # noqa: F401
    import tokenizers  # noqa: F401

    offline_fastembed()
    kept: list[Any] = []
    print(json.dumps({"key": "baseline", "mb": rss_mb()}))
    for key in keys:
        before = rss_mb()
        start = time.perf_counter()
        kept.append(load(key))
        print(json.dumps({"key": key, "load_s": time.perf_counter() - start, "mb": rss_mb() - before}))
    print(json.dumps({"key": "total", "mb": rss_mb()}))


def run_child(keys: Sequence[str]) -> list[dict[str, Any]]:
    out = subprocess.run(  # noqa: S603 - this module on the current interpreter
        [sys.executable, "-m", "tests.unit.guardrails.ml.test_bench_ml", *keys],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def ctx(text: str, *, stage: Any = "input", role: Any = "user", **request: Any) -> GuardContext:
    return GuardContext(
        stage=stage,
        request_id="bench",
        segments=(Segment(index=0, role=role, kind="content", msg=0, text=text),),
        request=make_request(**request),
        vault=GuardVault(),
    )


async def timed(
    check: Callable[[GuardContext], Awaitable[GuardFinding]], inputs: Sequence[tuple[str, GuardContext]]
) -> str:
    samples: dict[str, list[float]] = {}
    for round_ in range(WARMUP + REPEAT):
        for bucket, gctx in inputs:
            start = time.perf_counter()
            await check(gctx)
            if round_ >= WARMUP:
                samples.setdefault(bucket, []).append((time.perf_counter() - start) * 1000)
    every = [s for values in samples.values() for s in values]
    line = f"p50 {percentile(every, 0.5):6.1f} ms  p99 {percentile(every, 0.99):6.1f} ms"
    if len(samples) > 1:
        line += "  | p99 by chars " + " ".join(f"{b}:{percentile(v, 0.99):.0f}" for b, v in samples.items())
    return line


async def test_guard_models_memory_and_latency() -> None:
    keys = available()
    unit = "resident" if sys.platform == "linux" else "phys_footprint"
    rows = ["", f"memory ({unit}), fresh process per model:"]
    baseline: dict[str, Any] = {}
    for key in keys:
        baseline, entry, _ = run_child([key])
        rows.append(f"  {key:<40} load {entry['load_s']:5.2f} s  +{entry['mb']:6.0f} MB")
    every = run_child(keys)
    rows.append(f"  baseline after importing onnxruntime/tokenizers/presidio: {baseline['mb']:.0f} MB")
    rows.append(f"  all of them in one process: {every[-1]['mb']:.0f} MB total")
    missing = [m for m in (TOXICITY, HHEM) if m not in keys]
    rows.append(f"  missing: {', '.join(missing) or 'none'}")

    offline_fastembed()
    cpu = CpuExecutor(2, 64)
    texts = corpus()
    prompts = [(bucket, ctx(text)) for bucket, text in texts]
    windows = [
        ("250", ctx(t[i : i + 250], stage="output", role="assistant"))
        for _, t in texts[:12]
        for i in (0, 120)
    ]
    rows.append(f"latency per guard call via the cpu executor ({len(prompts)} prompts of <=200/1k/4k chars):")
    pii_cfg = PiiNerCfg(entities={**PiiNerCfg().entities, "PERSON": "redact"})
    pii = PiiNer(pii_cfg, Resource.ready("p", load("presidio")), cpu)
    rows.append(f"  pii_ner [presidio + en_core_web_sm]: {await timed(pii.check, prompts)}")
    if "bge-small" in keys:
        from gg.cache.embedders.fastembed import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(cpu)
        entry = next(g for g in DEFAULT_POLICY["input"]["guards"] if g["guard"] == "topic")
        cfg = TopicCfg.model_validate({"deny": entry["deny"]})
        topic = Topic(cfg, Resource.ready("i", await build_index(cfg, embedder)), embedder)
        rows.append(f"  topic [bge-small, shared embedder]: {await timed(topic.check, prompts)}")
    if TOXICITY in keys:
        tox = Toxicity(ToxicityCfg(), Resource.ready("t", load(TOXICITY)), cpu)
        rows.append(f"  toxicity [{TOXICITY}] 250-char stream window: {await timed(tox.check, windows)}")
        rows.append(f"  toxicity [{TOXICITY}] whole reply: {await timed(tox.check, prompts)}")
    if HHEM in keys:
        context = " ".join(text for _, text in texts[:4])[:1500]
        request = {"messages": [{"role": "system", "content": context}, {"role": "user", "content": "q"}]}
        reply = (
            "The pipeline runs unit tests first. Revenue grew last quarter. The office moved to Leeds in May."
        )
        ground = Grounding(GroundingCfg(), Resource.ready("h", load(HHEM)), cpu)
        reply_ctx = ctx(reply, stage="output", role="assistant", **request)
        rows.append(
            f"  grounding [hhem, 1.5k context, 3 claims]: {await timed(ground.check, [('r', reply_ctx)])}"
        )
    cpu.shutdown()
    print("\n".join(rows))


if __name__ == "__main__":
    measure(sys.argv[1:])
