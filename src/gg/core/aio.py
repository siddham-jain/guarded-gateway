import asyncio
import contextvars
import functools
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import anyio
import structlog

from gg.core.clock import Clock

log = structlog.get_logger("gg.aio")


class Deadline:
    def __init__(self, at_monotonic: float, clock: Clock) -> None:
        self._at = at_monotonic
        self._clock = clock

    @classmethod
    def after(cls, seconds: float, clock: Clock) -> "Deadline":
        return cls(clock.monotonic() + seconds, clock)

    @property
    def at(self) -> float:
        return self._at

    def remaining(self) -> float:
        return max(0.0, self._at - self._clock.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0

    def child(self, max_s: float) -> "Deadline":
        return Deadline(min(self._at, self._clock.monotonic() + max_s), self._clock)


async def run_shielded(fn: Callable[[], Awaitable[None]], *, name: str, timeout_s: float) -> None:
    # anyio re-cancels every await inside a cancelled scope, so asyncio.shield alone is not enough here
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(timeout_s) as scope:
            try:
                await fn()
            except Exception:
                log.exception("finalizer.failed", finalizer=name)
        if scope.cancelled_caught:
            log.warning("finalizer.timeout", finalizer=name, timeout_s=timeout_s)


class TaskSupervisor:
    """owns fire-and-forget tasks; keeps strong refs because asyncio only keeps weak ones"""

    def __init__(self, max_tasks: int = 1000) -> None:
        self._max = max_tasks
        self._tasks: set[asyncio.Task[None]] = set()
        self.rejected = 0

    def spawn(self, coro: Coroutine[Any, Any, None], *, name: str) -> bool:
        if len(self._tasks) >= self._max:
            coro.close()
            self.rejected += 1
            return False
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return True

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.error("task.failed", task=task.get_name(), error=repr(exc))

    def __len__(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout_s: float) -> None:
        if not self._tasks:
            return
        pending = set(self._tasks)
        _, still_pending = await asyncio.wait(pending, timeout=timeout_s)
        for task in still_pending:
            task.cancel()
        if still_pending:
            await asyncio.gather(*still_pending, return_exceptions=True)


class CpuSaturatedError(Exception):
    pass


class CpuExecutor:
    """bounded thread pool for cpu-bound work (onnx, presidio); separate from anyio's default limiter"""

    def __init__(self, workers: int, queue_max: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gg-cpu")
        self._capacity = workers + queue_max
        self._inflight = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    async def run[R](self, fn: Callable[..., R], *args: object) -> R:
        if self._inflight >= self._capacity:
            raise CpuSaturatedError("cpu executor saturated")
        self._inflight += 1
        try:
            ctx = contextvars.copy_context()
            call = functools.partial(ctx.run, fn, *args)
            return await asyncio.get_running_loop().run_in_executor(self._pool, call)
        finally:
            self._inflight -= 1

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
