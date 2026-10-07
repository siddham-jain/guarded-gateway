import asyncio

import pytest

from gg.core.aio import CpuExecutor, CpuSaturatedError, Deadline, TaskSupervisor
from gg.core.clock import FakeClock
from gg.core.context import ContextKey, FinalizerOrder, Finalizers
from tests.conftest import make_ctx


def test_deadline_child_never_extends_parent(clock: FakeClock) -> None:
    parent = Deadline.after(10, clock)
    assert parent.child(30).remaining() == 10
    assert parent.child(3).remaining() == 3
    clock.advance(11)
    assert parent.expired()
    assert parent.remaining() == 0


def test_stage_timings_server_timing(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    with ctx.timings.measure("auth"):
        clock.advance(0.002)
    ctx.timings.record("auth", 0.001)
    assert ctx.timings.server_timing() == "auth;dur=3.00"
    ctx.timings.mark("a")
    clock.advance(1)
    ctx.timings.mark("b")
    ctx.timings.mark("a")
    assert ctx.timings.between("a", "b") == 1


def test_context_extension_bag(clock: FakeClock) -> None:
    ctx = make_ctx(clock)
    key = ContextKey[int]("n")
    assert ctx.get(key) is None
    ctx.set(key, 3)
    assert ctx.get(key) == 3
    assert "hello" not in repr(ctx)


async def test_finalizers_run_once_in_order_and_survive_failures() -> None:
    calls: list[str] = []
    fin = Finalizers()

    async def record(name: str) -> None:
        calls.append(name)

    async def boom() -> None:
        raise RuntimeError("x")

    fin.defer("log", lambda: record("log"), FinalizerOrder.LOG)
    fin.defer("boom", boom, FinalizerOrder.LIMITS)
    fin.defer("budget", lambda: record("budget"), FinalizerOrder.BUDGET)
    await fin.run(timeout_s=1)
    await fin.run(timeout_s=1)
    assert calls == ["budget", "log"]


async def test_finalizer_runs_even_when_caller_is_cancelled() -> None:
    done = asyncio.Event()
    fin = Finalizers()

    async def slow() -> None:
        await asyncio.sleep(0.01)
        done.set()

    fin.defer("slow", slow)

    async def caller() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await fin.run(timeout_s=1)

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert done.is_set()


async def test_task_supervisor_caps_and_drains() -> None:
    sup = TaskSupervisor(max_tasks=1)
    gate = asyncio.Event()

    async def wait() -> None:
        await gate.wait()

    assert sup.spawn(wait(), name="a")
    assert not sup.spawn(wait(), name="b")
    assert sup.rejected == 1
    gate.set()
    await sup.drain(1)
    assert len(sup) == 0


async def test_cpu_executor_saturation() -> None:
    executor = CpuExecutor(workers=1, queue_max=0)
    try:
        assert await executor.run(sum, [1, 2]) == 3
        blocker = asyncio.Event()
        loop = asyncio.get_running_loop()

        def block() -> None:
            fut = asyncio.run_coroutine_threadsafe(blocker.wait(), loop)
            fut.result(timeout=2)

        first = asyncio.create_task(executor.run(block))
        await asyncio.sleep(0.01)
        with pytest.raises(CpuSaturatedError):
            await executor.run(sum, [1])
        blocker.set()
        await first
    finally:
        executor.shutdown()
