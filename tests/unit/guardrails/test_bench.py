"""latency micro-benchmark of the input pre phase (tiers 0/1) on a fixed corpus; -s shows the numbers"""

import random
import time

from gg.core.clock import FakeClock
from gg.guardrails.eval.stats import percentile
from gg.guardrails.fakes import FakeValues
from gg.guardrails.stages import InputGuardPreStage
from gg.pipeline.stage import PipelineResult
from tests.conftest import make_ctx, make_request
from tests.unit.guardrails.support import engine, policy_set

SENTENCES = (
    "The deployment pipeline runs the unit tests before the integration suite.",
    "Can you explain how a circuit breaker decides when to open again?",
    "Our quarterly revenue grew while support tickets dropped by a fifth.",
    "Please summarise the attached design document in five bullet points.",
    "Refactor this function so it no longer mutates its argument in place.",
    "What is the difference between a process and a thread on linux?",
)
BUCKETS = {"200": 200, "1k": 1_000, "4k": 4_000}
PER_BUCKET = 12
REPEAT = 20
WARMUP = 3


def corpus() -> list[tuple[str, str]]:
    rng = random.Random(42)  # noqa: S311 - fixed corpus, not security
    fakes = FakeValues(3)
    out: list[tuple[str, str]] = []
    for bucket, size in BUCKETS.items():
        for i in range(PER_BUCKET):
            words: list[str] = []
            while sum(len(w) + 1 for w in words) < size:
                words.append(rng.choice(SENTENCES))
            if i % 3 == 1:
                words.insert(len(words) // 2, f"Mail {fakes.get('email')} or call {fakes.get('phone')}.")
            if i % 3 == 2:
                words.insert(len(words) // 2, f"The token is {fakes.get('github_pat')}.")
            out.append((bucket, " ".join(words)[:size]))
    return out


def _p(values: list[float]) -> str:
    return f"p50={percentile(values, 0.5):.3f} ms p99={percentile(values, 0.99):.3f} ms"


async def _next(ctx: object) -> PipelineResult:
    return PipelineResult(source="upstream")


async def test_input_pre_phase_latency() -> None:
    stage = InputGuardPreStage(policy_set(), engine())
    samples: dict[str, list[float]] = {b: [] for b in BUCKETS}
    clock = FakeClock()
    for round_ in range(WARMUP + REPEAT):
        for bucket, text in corpus():
            ctx = make_ctx(clock, make_request(messages=[{"role": "user", "content": text}]))
            start = time.perf_counter()
            await stage(ctx, _next)
            elapsed = (time.perf_counter() - start) * 1000
            if round_ >= WARMUP:
                samples[bucket].append(elapsed)
    every = [s for bucket in samples.values() for s in bucket]
    print(f"\ninput pre phase over {len(every)} runs: {_p(every)}")
    for bucket, values in samples.items():
        print(f"  <= {bucket:<3} chars: {_p(values)}")
    # generous ceiling only (C6 §8.5); shared runners are noisy
    assert percentile(every, 0.99) < 100
