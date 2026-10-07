"""bge-small via fastembed (onnx) on the shared cpu executor; needs the `ml` extra, imported on first use"""

import asyncio
import importlib
import importlib.util
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gg.core.aio import CpuExecutor


class FastEmbedEmbedder:
    def __init__(
        self,
        cpu: CpuExecutor,
        *,
        model: str = "BAAI/bge-small-en-v1.5",
        dim: int = 384,
        cache_dir: Path | None = None,
    ) -> None:
        self._cpu = cpu
        self._model_name = model
        self._dim = dim
        self._cache_dir = cache_dir
        self._model: Any = None
        self._loading = asyncio.Lock()

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("fastembed") is not None

    @property
    def name(self) -> str:
        return self._model_name.rsplit("/", 1)[-1]

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> Any:
        try:
            module = importlib.import_module("fastembed")
        except ImportError as exc:
            raise RuntimeError("the fastembed embedder needs the 'ml' extra (uv sync --extra ml)") from exc
        cache_dir = str(self._cache_dir) if self._cache_dir is not None else None
        return module.TextEmbedding(model_name=self._model_name, threads=1, cache_dir=cache_dir)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for raw in self._model.embed(texts):
            vector = [float(x) for x in raw]
            if len(vector) != self._dim:
                raise RuntimeError(f"{self._model_name} returned dim {len(vector)}, config says {self._dim}")
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            out.append([x / norm for x in vector])
        return out

    async def start(self) -> None:
        async with self._loading:
            if self._model is None:
                self._model = await self._cpu.run(self._load)

    async def embed(self, texts: Sequence[str], /) -> list[list[float]]:
        if self._model is None:
            await self.start()
        return await self._cpu.run(self._embed, list(texts))
