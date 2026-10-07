import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from gg.core.aio import CpuExecutor
from gg.guardrails.ml.artefacts import MODELS, ArtefactStore
from tests.unit.guardrails.support import ROOT

MODELS_DIR = ROOT / ".models"
FASTEMBED_DIR = MODELS_DIR / "fastembed"


def store() -> ArtefactStore:
    return ArtefactStore(MODELS_DIR, download=False)


def needs_weights(*model_ids: str) -> pytest.MarkDecorator:
    missing = [m for m in model_ids if not store().present(MODELS[m])]
    return pytest.mark.skipif(bool(missing), reason=f"weights not in {MODELS_DIR}: {', '.join(missing)}")


def needs_fastembed_cache() -> pytest.MarkDecorator:
    present = any(FASTEMBED_DIR.glob("models--Qdrant--bge-small-en-v1.5*"))
    return pytest.mark.skipif(not present, reason=f"bge-small not cached in {FASTEMBED_DIR}")


@dataclass
class FakeClassifier:
    """scores by substring: the highest score of any key found in the text, else `default`"""

    by_substring: dict[str, float] = field(default_factory=lambda: {})
    default: float = 0.01
    seen: list[str] = field(default_factory=lambda: [])

    def scores(self, texts: Sequence[str], /, *, max_windows: int = 16) -> list[float]:
        self.seen.extend(texts)
        return [max((s for k, s in self.by_substring.items() if k in t), default=self.default) for t in texts]


class FakeEmbedder:
    """bag of words over a fixed vocabulary; unit length like the real embedder"""

    name = "fake-embedder"
    dim = 8

    def __init__(self, vocab: Sequence[str]) -> None:
        self._vocab = list(vocab)
        self.calls = 0

    async def embed(self, texts: Sequence[str], /) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for text in texts:
            words = text.lower().split()
            vec = [float(sum(w.startswith(v) for w in words)) for v in self._vocab]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


@dataclass(frozen=True)
class Hit:
    entity: str
    start: int
    end: int
    score: float


class FakeAnalyzer:
    def __init__(self, find: Callable[[str], list[Hit]]) -> None:
        self._find = find

    def analyze(self, text: str, entities: Sequence[str], min_score: float, /) -> list[Hit]:
        return [h for h in self._find(text) if h.entity in entities and h.score >= min_score]


def cpu() -> CpuExecutor:
    return CpuExecutor(2, 8)
