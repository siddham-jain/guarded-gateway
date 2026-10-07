import random
from collections.abc import Callable
from dataclasses import dataclass

from gg.core.clock import FakeClock
from gg.core.errors import GGError
from gg.reliability.policy import RetryConfig
from tests.unit.reliability.fakes import (
    FakeAdapter,
    Harness,
    Step,
    context_for,
    deployment,
    drain,
    ok,
    overloaded,
)

REQUESTS = 200
PRIMARY = deployment("primary/model")
SECONDARY = deployment("secondary/model")


def flaky(name: str, failure_rate: float, rng: random.Random) -> Callable[[], Step]:
    def next_step() -> Step:
        return overloaded() if rng.random() < failure_rate else ok(f"{name}:", "answer")

    return next_step


@dataclass
class Run:
    successes: int = 0
    failures: int = 0
    max_attempts: int = 0


async def run_requests(h: Harness, clock: FakeClock, rng: random.Random) -> Run:
    run = Run()
    for _ in range(REQUESTS):
        stream = rng.random() < 0.5
        ctx = context_for(clock, [PRIMARY, SECONDARY], stream=stream)
        try:
            result = await h.executor(ctx)
            if result.stream is not None:
                text = await drain(result.stream)
            else:
                assert result.response is not None
                text = result.response.choices[0].message.content or ""
        except GGError:
            run.failures += 1
        else:
            # provenance: the whole answer comes from the one deployment that served it
            assert ctx.served_by is not None
            assert text == f"{ctx.served_by.provider}:answer"
            run.successes += 1
        run.max_attempts = max(run.max_attempts, len(ctx.attempts))
        clock.advance(0.5)
    return run


async def test_half_failing_primary_with_healthy_secondary(clock: FakeClock) -> None:
    rng = random.Random(42)  # noqa: S311
    adapters = {
        "primary": FakeAdapter("primary", next_step=flaky("primary", 0.5, rng)),
        "secondary": FakeAdapter("secondary", next_step=flaky("secondary", 0.0, rng)),
    }
    h = Harness(clock, adapters)

    run = await run_requests(h, clock, rng)

    assert run.successes / REQUESTS >= 0.99
    cfg = RetryConfig()
    assert run.max_attempts <= (cfg.max_hops + 1) * (cfg.max_retries + 1)
    assert adapters["primary"].calls > 0
    assert adapters["primary"].closed == adapters["primary"].calls
    assert adapters["secondary"].closed == adapters["secondary"].calls


async def test_dead_primary_is_cut_off_by_breaker(clock: FakeClock) -> None:
    rng = random.Random(7)  # noqa: S311
    adapters = {
        "primary": FakeAdapter("primary", next_step=flaky("primary", 1.0, rng)),
        "secondary": FakeAdapter("secondary", next_step=flaky("secondary", 0.0, rng)),
    }
    h = Harness(clock, adapters)

    run = await run_requests(h, clock, rng)

    assert run.successes == REQUESTS
    # 3 to trip, then one probe per cooldown (30 s, 60 s) over the 100 s of fake time
    assert adapters["primary"].calls <= 5
    assert h.breakers.get(PRIMARY).state == "open"
