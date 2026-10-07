"""real-weight checks; skipped unless the artefacts are in .models/ (CI without weights skips them)"""

from collections.abc import AsyncIterator
from functools import cache
from typing import Any

import pytest

from gg.core.aio import CpuExecutor
from gg.core.clock import SystemClock
from gg.core.guard_types import Verdict
from gg.guardrails.base import GuardContext, Segment
from gg.guardrails.ml.artefacts import MODELS
from gg.guardrails.ml.common import PairScorer, TextClassifier
from gg.guardrails.ml.grounding import Grounding, GroundingCfg
from gg.guardrails.ml.pii_ner import Analyzer, PiiNer, PiiNerCfg
from gg.guardrails.ml.runtime import Resource
from gg.guardrails.ml.topic import Topic, TopicCfg, build_index
from gg.guardrails.ml.toxicity import Toxicity, ToxicityCfg
from gg.guardrails.setup import build_guardrails
from gg.guardrails.vault import GuardVault
from gg.pipeline.stage import PipelineResult
from tests.conftest import make_request
from tests.unit.guardrails.ml.support import (
    FASTEMBED_DIR,
    MODELS_DIR,
    FakeEmbedder,
    needs_fastembed_cache,
    needs_weights,
    store,
)
from tests.unit.guardrails.support import DEFAULT_POLICY, POLICY_DIR

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

TOXICITY = "granite-guardian-hap-38m"
HHEM = "hhem-2.1-open"


@pytest.fixture
async def cpu() -> AsyncIterator[CpuExecutor]:
    executor = CpuExecutor(2, 8)
    yield executor
    executor.shutdown()


@cache
def classifier(model_id: str) -> TextClassifier:
    from gg.guardrails.ml.onnx import OnnxTextClassifier

    return OnnxTextClassifier.load(MODELS[model_id], store().fetch(MODELS[model_id]))


@cache
def presidio() -> Analyzer:
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("en_core_web_sm")
    from gg.guardrails.ml.presidio import PresidioAnalyzer

    return PresidioAnalyzer("en_core_web_sm")


@cache
def hhem() -> PairScorer:
    from gg.guardrails.ml.hhem import HhemScorer

    return HhemScorer(MODELS[HHEM], store().fetch(MODELS[HHEM]))


def gctx(text: str, *, stage: Any = "input", role: Any = "user", **request: Any) -> GuardContext:
    return GuardContext(
        stage=stage,
        request_id="req_w",
        segments=(Segment(index=0, role=role, kind="content", msg=0, text=text),),
        request=make_request(**request),
        vault=GuardVault(),
    )


async def test_presidio_finds_person_phone_and_date_of_birth(cpu: CpuExecutor) -> None:
    cfg = PiiNerCfg(entities={"PERSON": "redact", "PHONE_NUMBER": "redact", "DATE_OF_BIRTH": "redact"})
    guard = PiiNer(cfg, Resource.ready("p", presidio()), cpu)
    text = "I'm Jane Morrison, born on 4 May 1990. My phone number is +44 20 7946 0958."
    found = await guard.check(gctx(text))
    assert found.verdict is Verdict.REDACT
    assert set(found.labels) == {"PERSON", "PHONE", "DOB"}
    values = {r.label: text[r.start : r.end] for r in found.redactions}
    assert values == {"PERSON": "Jane Morrison", "PHONE": "+44 20 7946 0958", "DOB": "4 May 1990"}


@needs_weights(TOXICITY)
async def test_toxic_output_is_flagged_and_polite_output_allowed(cpu: CpuExecutor) -> None:
    guard = Toxicity(ToxicityCfg(), Resource.ready("t", classifier(TOXICITY)), cpu)
    toxic = await guard.check(gctx("You are a worthless idiot and everyone hates you.", stage="output"))
    polite = await guard.check(gctx("Thanks for asking, here is the summary you wanted.", stage="output"))
    assert toxic.verdict >= Verdict.FLAG
    assert polite.verdict is Verdict.ALLOW


@needs_fastembed_cache()
async def test_topic_matches_a_deny_topic_with_the_shared_embedder(
    cpu: CpuExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastembed")
    from gg.cache.embedders.fastembed import FastEmbedEmbedder

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(FASTEMBED_DIR))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    entry = next(g for g in DEFAULT_POLICY["input"]["guards"] if g["guard"] == "topic")
    cfg = TopicCfg.model_validate({"deny": entry["deny"]})
    embedder = FastEmbedEmbedder(cpu)
    guard = Topic(cfg, Resource.ready("i", await build_index(cfg, embedder)), embedder)
    hit = await guard.check(gctx("What's the maximum dose of ibuprofen I can give my 5 year old?"))
    miss = await guard.check(gctx("How do I bake sourdough bread?"))
    assert hit.verdict is Verdict.BLOCK
    assert hit.reason == "medical_dosage"
    assert miss.verdict is Verdict.ALLOW


@needs_weights(HHEM)
async def test_grounding_flags_a_claim_the_context_does_not_support(cpu: CpuExecutor) -> None:
    guard = Grounding(GroundingCfg(min_context_chars=50), Resource.ready("h", hhem()), cpu)
    context = (
        "The Eiffel Tower is a wrought-iron tower in Paris. It was completed in 1889 for the World's Fair."
    )
    request = {"messages": [{"role": "system", "content": context}, {"role": "user", "content": "Tell me."}]}
    good = await guard.check(
        gctx("The Eiffel Tower was completed in 1889.", stage="output", role="assistant", **request)
    )
    bad = await guard.check(
        gctx("The Eiffel Tower was built in Berlin in 1920.", stage="output", role="assistant", **request)
    )
    assert good.verdict is Verdict.ALLOW
    assert bad.verdict is Verdict.FLAG


async def _next(ctx: Any) -> PipelineResult:
    return PipelineResult(source="upstream")


@needs_weights(TOXICITY)
async def test_default_policy_models_load_and_report_healthy(cpu: CpuExecutor) -> None:
    pytest.importorskip("presidio_analyzer")
    guardrails = build_guardrails(
        POLICY_DIR,
        clock=SystemClock(),
        cpu=cpu,
        embedder=FakeEmbedder(["dose", "gun"]),
        models_dir=MODELS_DIR,
        download_models=False,
    )
    await guardrails.start()
    assert (await guardrails.health.check()).status == "ok"
