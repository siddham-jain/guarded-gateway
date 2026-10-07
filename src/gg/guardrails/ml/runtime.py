"""shared model runtime: one load per model across all effective policies, readiness, load stats"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import structlog

from gg.cache.base import Embedder
from gg.core.aio import CpuExecutor
from gg.core.lifecycle import HealthStatus
from gg.guardrails.ml.artefacts import ArtefactStore, model_spec
from gg.guardrails.ml.common import PairScorer, TextClassifier

log = structlog.get_logger("gg.guardrails.ml")


class Resource[T]:
    """loaded once, in its own task, so a guard timing out while waiting never restarts the load"""

    def __init__(self, name: str, load: Callable[[], Awaitable[T]], *, lazy: bool = False) -> None:
        self.name = name
        self.lazy = lazy
        self._load = load
        self._task: asyncio.Task[T] | None = None
        self.load_s: float | None = None

    @classmethod
    def ready(cls, name: str, value: T) -> "Resource[T]":
        async def load() -> T:
            return value

        return cls(name, load)

    async def _timed(self) -> T:
        start = time.perf_counter()
        value = await self._load()
        self.load_s = time.perf_counter() - start
        log.info("guard_model.loaded", model=self.name, load_s=round(self.load_s, 3))
        return value

    async def get(self) -> T:
        if self._task is None:
            self._task = asyncio.create_task(self._timed(), name=f"guard-model-{self.name}")
        return await asyncio.shield(self._task)

    @property
    def loaded(self) -> bool:
        task = self._task
        return task is not None and task.done() and not task.cancelled() and task.exception() is None

    @property
    def error(self) -> BaseException | None:
        task = self._task
        if task is None or not task.done() or task.cancelled():
            return None
        return task.exception()


@dataclass(frozen=True, slots=True)
class LoadReport:
    name: str
    loaded: bool
    load_s: float | None
    error: str | None


class MlRuntime:
    """owns every model-backed resource; the composition root starts it and gates readiness on it"""

    def __init__(self, cpu: CpuExecutor, store: ArtefactStore, embedder: Embedder | None = None) -> None:
        self.cpu = cpu
        self.store = store
        self.embedder = embedder
        self._resources: dict[str, Resource[Any]] = {}
        self._started = False

    def resource[T](self, key: str, load: Callable[[], Awaitable[T]], *, lazy: bool = False) -> Resource[T]:
        existing = self._resources.get(key)
        if existing is None:
            existing = self._resources[key] = Resource(key, load, lazy=lazy)
        return cast("Resource[T]", existing)

    def classifier(self, model_id: str, *, lazy: bool = False) -> Resource[TextClassifier]:
        spec = model_spec(model_id)

        def load() -> TextClassifier:
            from gg.guardrails.ml.onnx import OnnxTextClassifier

            return OnnxTextClassifier.load(spec, self.store.fetch(spec))

        async def run() -> TextClassifier:
            return await self.cpu.run(load)

        return self.resource(f"model:{model_id}", run, lazy=lazy)

    def pair_scorer(self, model_id: str, *, lazy: bool = True) -> Resource[PairScorer]:
        spec = model_spec(model_id)

        def load() -> PairScorer:
            from gg.guardrails.ml.hhem import HhemScorer

            return HhemScorer(spec, self.store.fetch(spec))

        async def run() -> PairScorer:
            return await self.cpu.run(load)

        return self.resource(f"model:{model_id}", run, lazy=lazy)

    async def start(self) -> None:
        """loads and warms every non-lazy resource in turn; failures are logged and keep readiness down"""
        for res in list(self._resources.values()):
            if res.lazy:
                continue
            try:
                await res.get()
            except Exception as exc:
                log.error("guard_model.load_failed", model=res.name, error=repr(exc))
        self._started = True

    def report(self) -> list[LoadReport]:
        return [
            LoadReport(r.name, r.loaded, r.load_s, None if r.error is None else repr(r.error))
            for r in self._resources.values()
        ]

    async def check(self) -> HealthStatus:
        required = [r for r in self._resources.values() if not r.lazy]
        failed = sorted(r.name for r in required if r.error is not None)
        if failed:
            return HealthStatus("guard_models", "down", f"failed: {', '.join(failed)}")
        pending = sorted(r.name for r in required if not r.loaded)
        if pending or not self._started:
            return HealthStatus("guard_models", "down", f"loading: {', '.join(pending)}")
        return HealthStatus("guard_models", "ok", f"{len(required)} loaded")
