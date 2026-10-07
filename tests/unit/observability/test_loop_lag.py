import asyncio

import pytest

from gg.core.clock import FakeClock, SystemClock
from gg.core.lifecycle import Lifecycle
from gg.observability.loop_lag import EventLoopLagSampler
from tests.unit.observability.support import new_metrics


async def test_sample_records_wakeup_lateness(clock: FakeClock) -> None:
    metrics = new_metrics()

    async def late_sleep(seconds: float) -> None:
        clock.advance(seconds + 0.03)

    sampler = EventLoopLagSampler(metrics, clock=clock, sleep=late_sleep)
    assert await sampler.sample() == pytest.approx(0.03)
    assert metrics.registry.get_sample_value("gg_event_loop_lag_seconds_sum") == pytest.approx(0.03)
    assert metrics.registry.get_sample_value("gg_event_loop_lag_seconds_bucket", {"le": "0.05"}) == 1


async def test_early_wakeup_is_zero_lag(clock: FakeClock) -> None:
    async def early_sleep(seconds: float) -> None:
        clock.advance(seconds - 0.001)

    assert await EventLoopLagSampler(new_metrics(), clock=clock, sleep=early_sleep).sample() == 0.0


async def test_start_stop_lifecycle() -> None:
    metrics = new_metrics()
    sampler = EventLoopLagSampler(metrics, clock=SystemClock(), interval_s=0.001)
    assert isinstance(sampler, Lifecycle)
    await sampler.start()
    await sampler.start()
    assert sampler.running
    for _ in range(100):
        if metrics.registry.get_sample_value("gg_event_loop_lag_seconds_count"):
            break
        await asyncio.sleep(0.005)
    await sampler.stop()
    await sampler.stop()
    assert not sampler.running
    assert (metrics.registry.get_sample_value("gg_event_loop_lag_seconds_count") or 0) >= 1


def test_rejects_non_positive_interval(clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="positive"):
        EventLoopLagSampler(new_metrics(), clock=clock, interval_s=0)
