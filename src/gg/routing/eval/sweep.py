"""offline alpha sweep and routellm metrics (C5 §10.7) over cached grades and costs; pure arithmetic.

a router with scores s routes item i strong iff s_i >= alpha. sorting by score and walking tie blocks gives
the cost-quality curve; inside a tie block the curve is linear, which is the expectation of random
tie-breaking.
"""

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

PGR_BINS = tuple(round(0.1 * i, 1) for i in range(1, 11))


@dataclass(frozen=True, slots=True)
class Outcome:
    item_id: str
    category: str
    split: str
    g_weak: float
    g_strong: float
    cost_weak: float
    cost_strong: float

    @property
    def strong_wins(self) -> bool:
        return self.g_strong > self.g_weak


@dataclass(frozen=True, slots=True)
class CurvePoint:
    strong_share: float
    quality: float
    cost: float
    # the alpha that yields this point; inf routes nothing strong
    alpha: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def endpoints(outcomes: Sequence[Outcome]) -> tuple[float, float]:
    """(always-weak quality, always-strong quality)"""
    return _mean([o.g_weak for o in outcomes]), _mean([o.g_strong for o in outcomes])


def curve(
    outcomes: Sequence[Outcome], scores: Mapping[str, float], extra_cost: float = 0.0
) -> list[CurvePoint]:
    n = len(outcomes)
    if n == 0:
        return []
    ranked = sorted(outcomes, key=lambda o: scores[o.item_id], reverse=True)
    quality = sum(o.g_weak for o in outcomes)
    cost = sum(o.cost_weak for o in outcomes)
    points = [CurvePoint(0.0, quality / n, cost / n + extra_cost, math.inf)]
    i = 0
    while i < n:
        alpha = scores[ranked[i].item_id]
        while i < n and scores[ranked[i].item_id] == alpha:
            o = ranked[i]
            quality += o.g_strong - o.g_weak
            cost += o.cost_strong - o.cost_weak
            i += 1
        points.append(CurvePoint(i / n, quality / n, cost / n + extra_cost, alpha))
    return points


def interpolate(points: Sequence[CurvePoint], share: float) -> tuple[float, float]:
    """(quality, cost) at a strong share on the piecewise-linear curve"""
    share = min(1.0, max(0.0, share))
    for left, right in pairwise(points):
        if left.strong_share <= share <= right.strong_share:
            width = right.strong_share - left.strong_share
            t = 0.0 if width == 0 else (share - left.strong_share) / width
            return (
                left.quality + t * (right.quality - left.quality),
                left.cost + t * (right.cost - left.cost),
            )
    last = points[-1]
    return last.quality, last.cost


def pgr(quality: float, q_weak: float, q_strong: float) -> float | None:
    gap = q_strong - q_weak
    return None if gap <= 0 else (quality - q_weak) / gap


def apgr_trapz(points: Sequence[CurvePoint], q_weak: float, q_strong: float) -> float | None:
    """routellm code form: (auc_router - auc_weak) / (auc_strong - auc_weak) over strong share in [0, 1]"""
    gap = q_strong - q_weak
    if gap <= 0 or not points:
        return None
    auc = sum((b.strong_share - a.strong_share) * (a.quality + b.quality) / 2 for a, b in pairwise(points))
    return (auc - q_weak) / gap


def apgr_bins(points: Sequence[CurvePoint], q_weak: float, q_strong: float) -> float | None:
    """routellm paper form: mean pgr at 10%, 20%, ..., 100% strong calls"""
    values = [pgr(interpolate(points, share)[0], q_weak, q_strong) for share in PGR_BINS]
    if any(v is None for v in values):
        return None
    return _mean([v for v in values if v is not None])


def cpt(points: Sequence[CurvePoint], target: float, q_weak: float, q_strong: float) -> float | None:
    """smallest strong share whose pgr reaches target on the interpolated curve"""
    if q_strong - q_weak <= 0:
        return None

    def at(p: CurvePoint) -> float:
        return (p.quality - q_weak) / (q_strong - q_weak)

    if at(points[0]) >= target:
        return points[0].strong_share
    for left, right in pairwise(points):
        lo, hi = at(left), at(right)
        if hi >= target > lo:
            t = (target - lo) / (hi - lo)
            return left.strong_share + t * (right.strong_share - left.strong_share)
    return None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """mann-whitney u with average ranks for ties"""
    pairs = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += average_rank * sum(1 for p in pairs[i:j] if p[1])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, centre + half)


@dataclass(frozen=True, slots=True)
class AtAlpha:
    alpha: float
    strong_share: float
    quality: float
    pgr: float | None
    cost: float
    under_routed: int
    strong_wins: int

    @property
    def under_route_rate(self) -> float | None:
        return self.under_routed / self.strong_wins if self.strong_wins else None


def at_alpha(
    outcomes: Sequence[Outcome], scores: Mapping[str, float], alpha: float, extra_cost: float = 0.0
) -> AtAlpha:
    n = len(outcomes)
    strong = [scores[o.item_id] >= alpha for o in outcomes]
    quality = _mean([o.g_strong if s else o.g_weak for o, s in zip(outcomes, strong, strict=True)])
    cost = _mean([o.cost_strong if s else o.cost_weak for o, s in zip(outcomes, strong, strict=True)])
    q_weak, q_strong = endpoints(outcomes)
    wins = [o.strong_wins for o in outcomes]
    return AtAlpha(
        alpha=alpha,
        strong_share=sum(strong) / n if n else 0.0,
        quality=quality,
        pgr=pgr(quality, q_weak, q_strong),
        cost=cost + extra_cost,
        under_routed=sum(1 for w, s in zip(wins, strong, strict=True) if w and not s),
        strong_wins=sum(wins),
    )


def alpha_for_pgr(
    points: Sequence[CurvePoint], target: float, q_weak: float, q_strong: float
) -> float | None:
    """highest alpha (fewest strong calls) whose discrete operating point reaches the target pgr"""
    for p in points:
        value = pgr(p.quality, q_weak, q_strong)
        if value is not None and value >= target:
            return p.alpha
    return None


def random_curve(outcomes: Sequence[Outcome]) -> list[CurvePoint]:
    """expected curve of a router that sends a random share strong: the straight line between endpoints"""
    q_weak, q_strong = endpoints(outcomes)
    c_weak = _mean([o.cost_weak for o in outcomes])
    c_strong = _mean([o.cost_strong for o in outcomes])
    return [CurvePoint(0.0, q_weak, c_weak, math.inf), CurvePoint(1.0, q_strong, c_strong, 0.0)]


def oracle_scores(outcomes: Sequence[Outcome]) -> dict[str, float]:
    """strong exactly where it helps most: the upper bound for any router at every budget"""
    return {o.item_id: o.g_strong - o.g_weak for o in outcomes}


type Statistic = Callable[[Sequence[Outcome]], float | None]


def bootstrap_ci(
    outcomes: Sequence[Outcome], statistic: Statistic, *, resamples: int = 1000, seed: int = 7
) -> tuple[float, float] | None:
    """95% percentile interval; resamples where the statistic is undefined (no gap) are dropped"""
    if not outcomes:
        return None
    rng = random.Random(seed)  # noqa: S311
    n = len(outcomes)
    values: list[float] = []
    for _ in range(resamples):
        sample = [outcomes[rng.randrange(n)] for _ in range(n)]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if len(values) < resamples // 2:
        return None
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[math.ceil(0.975 * (len(values) - 1))]
    return low, high
