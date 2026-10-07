"""pre-run cost estimate and the per-call hard cap (C11 §3.3 spend guard)"""

import math
from dataclasses import dataclass


class BudgetExceeded(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Price:
    """usd per 1m tokens at list price; billed is false for free tiers, which never count against the cap"""

    input: float
    output: float
    billed: bool = True

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


# chat template overhead per message, roughly what tiktoken-based counters add
MESSAGE_OVERHEAD_TOKENS = 4
# share of max_tokens assumed for the expected estimate; the worst case assumes every call hits the cap
EXPECTED_OUTPUT_SHARE = 0.5


def estimate_tokens(chars: int, messages: int = 1) -> int:
    return math.ceil(chars / 4) + MESSAGE_OVERHEAD_TOKENS * messages


@dataclass(frozen=True, slots=True)
class CallEstimate:
    model: str
    input_tokens: int
    max_output_tokens: int
    price: Price

    @property
    def worst_usd(self) -> float:
        return self.price.cost(self.input_tokens, self.max_output_tokens)

    @property
    def expected_usd(self) -> float:
        return self.price.cost(self.input_tokens, round(self.max_output_tokens * EXPECTED_OUTPUT_SHARE))

    @property
    def billed_worst_usd(self) -> float:
        return self.worst_usd if self.price.billed else 0.0


@dataclass(frozen=True, slots=True)
class Estimate:
    calls: int
    expected_usd: float
    worst_usd: float
    billed_expected_usd: float
    billed_worst_usd: float
    by_model: dict[str, tuple[int, float, float]]

    @classmethod
    def of(cls, calls: list[CallEstimate], extra_billed_usd: float = 0.0) -> "Estimate":
        by_model: dict[str, tuple[int, float, float]] = {}
        for c in calls:
            n, exp, worst = by_model.get(c.model, (0, 0.0, 0.0))
            by_model[c.model] = (n + 1, exp + c.expected_usd, worst + c.worst_usd)
        billed = [c for c in calls if c.price.billed]
        return cls(
            calls=len(calls),
            expected_usd=sum(c.expected_usd for c in calls) + extra_billed_usd,
            worst_usd=sum(c.worst_usd for c in calls) + extra_billed_usd,
            billed_expected_usd=sum(c.expected_usd for c in billed) + extra_billed_usd,
            billed_worst_usd=sum(c.worst_usd for c in billed) + extra_billed_usd,
            by_model=by_model,
        )

    def lines(self) -> list[str]:
        out = [
            f"estimate: {self.calls} uncached calls, list price expected ${self.expected_usd:.4f} "
            f"(worst ${self.worst_usd:.4f}); billed expected ${self.billed_expected_usd:.4f} "
            f"(worst ${self.billed_worst_usd:.4f})"
        ]
        out += [
            f"  {model:<40} {n:>4} calls  expected ${exp:.4f}  worst ${worst:.4f}"
            for model, (n, exp, worst) in sorted(self.by_model.items())
        ]
        return out


class SpendCap:
    """reserves a call's worst-case billed cost before it starts and settles the actual cost after"""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self._held_usd = 0.0

    def reserve(self, worst_usd: float) -> float:
        if self.spent_usd + self._held_usd + worst_usd > self.max_usd:
            raise BudgetExceeded(
                f"spend cap ${self.max_usd:.2f} reached (spent ${self.spent_usd:.4f}, "
                f"in flight ${self._held_usd:.4f}, next call up to ${worst_usd:.4f})"
            )
        self._held_usd += worst_usd
        return worst_usd

    def settle(self, held_usd: float, actual_usd: float) -> None:
        self._held_usd -= held_usd
        self.spent_usd += actual_usd
