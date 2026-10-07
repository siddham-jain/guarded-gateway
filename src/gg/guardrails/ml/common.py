"""ports the model-backed guards depend on, plus the shared threshold rule"""

from collections.abc import Sequence
from typing import Protocol

from gg.core.guard_types import Verdict

# reason on findings of model guards built without an ml runtime (the composition root passed no cpu executor)
ML_DISABLED = "ml_disabled"


class TextClassifier(Protocol):
    def scores(self, texts: Sequence[str], /, *, max_windows: int = 16) -> list[float]: ...


class PairScorer(Protocol):
    def scores(self, premise: str, hypotheses: Sequence[str], /) -> list[float]: ...


def graded(score: float, *, block_at: float, flag_at: float) -> Verdict:
    if score >= block_at:
        return Verdict.BLOCK
    if score >= flag_at:
        return Verdict.FLAG
    return Verdict.ALLOW


def check_thresholds(block_at: float, flag_at: float) -> None:
    if flag_at > block_at:
        raise ValueError("flag_at must not exceed block_at")
