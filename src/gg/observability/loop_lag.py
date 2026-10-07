import asyncio
from collections.abc import Awaitable, Callable

import structlog

from gg.core.clock import Clock
from gg.observability.metrics import Metrics

log = structlog.get_logger("gg.observability")


class EventLoopLagSampler:
    """sleeps interval_s in a loop and records how late each wake-up was"""

    def __init__(
        self,
        metrics: Metrics,
        *,
        clock: Clock,
        interval_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._metrics = metrics
        self._clock = clock
        self._interval = interval_s
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name="gg.loop_lag")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise

    async def sample(self) -> float:
        start = self._clock.monotonic()
        await self._sleep(self._interval)
        lag = max(0.0, self._clock.monotonic() - start - self._interval)
        self._metrics.observe_loop_lag(lag)
        return lag

    async def _run(self) -> None:
        while True:
            try:
                await self.sample()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._metrics.telemetry_error("loop_lag")
                log.exception("loop_lag.sample_failed")
