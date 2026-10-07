"""threshold sweep (C8 §8.2): pair distances via the production tags and index code, then per-τ counts"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from gg.cache.base import Embedder, SemanticTags
from gg.cache.eval.pairs import Pair
from gg.cache.keys import number_signature
from gg.cache.memory import InMemoryIndex
from gg.core.clock import SystemClock

_NO_SIG = "-"


@dataclass(frozen=True, slots=True)
class Scored:
    pair: Pair
    # None when the tag pre-filter (num_sig) already rules the candidate out
    distance: float | None


def thresholds(start: float = 0.0, stop: float = 0.30, step: float = 0.005) -> list[float]:
    count = round((stop - start) / step)
    return [round(start + i * step, 6) for i in range(count + 1)]


def _tags(pair: Pair, text: str, num_sig: bool) -> SemanticTags:
    # every pair gets its own scope so pairs never match each other
    sig = number_signature(text) if num_sig else _NO_SIG
    return SemanticTags(pair.id, "a", "r", "p", "s", "q", sig)


async def score_pairs(pairs: Sequence[Pair], embedder: Embedder, *, num_sig: bool) -> list[Scored]:
    index = InMemoryIndex(SystemClock())
    vectors = await embedder.embed([t for p in pairs for t in (p.anchor, p.candidate)])
    out: list[Scored] = []
    for i, pair in enumerate(pairs):
        anchor, candidate = vectors[2 * i], vectors[2 * i + 1]
        await index.add(anchor, _tags(pair, pair.anchor, num_sig), pair.id, 3600)
        match = await index.search(candidate, _tags(pair, pair.candidate, num_sig))
        out.append(Scored(pair, None if match is None else match.distance))
    return out


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def confusion(scored: Sequence[Scored], tau: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    false_hits: dict[str, int] = {}
    for s in scored:
        hit = s.distance is not None and s.distance <= tau
        if s.pair.should_hit:
            tp, fn = (tp + 1, fn) if hit else (tp, fn + 1)
        elif hit:
            fp += 1
            false_hits[s.pair.category] = false_hits.get(s.pair.category, 0) + 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    return {
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "false_hit_rate": None if precision is None else 1 - precision,
        "hit_rate": _ratio(tp, tp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_hits_by_category": false_hits,
    }


def choose_tau(rows: Sequence[dict[str, Any]], target_precision: float) -> float | None:
    """largest τ whose precision meets the target; τ with no predicted hits doesn't qualify"""
    ok = [r["tau"] for r in rows if r["precision"] is not None and r["precision"] >= target_precision]
    return max(ok) if ok else None


async def sweep(
    pairs: Sequence[Pair],
    embedder: Embedder,
    *,
    taus: Sequence[float],
    target_precision: float = 0.98,
    choose_on: str = "dev",
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, num_sig in (("embedding", False), ("embedding+num_sig", True)):
        scored = await score_pairs(pairs, embedder, num_sig=num_sig)
        tuning = [s for s in scored if s.pair.split == choose_on]
        held_out = [s for s in scored if s.pair.split != choose_on]
        tau_star = choose_tau([confusion(tuning, t) for t in taus], target_precision)
        variants[name] = {
            "tau_star": tau_star,
            "held_out_at_tau_star": None if tau_star is None else confusion(held_out, tau_star),
            "sweep": [confusion(scored, t) for t in taus],
            "distances": {s.pair.id: s.distance for s in scored},
        }
    return {
        "embedder": {"name": embedder.name, "dim": embedder.dim},
        "pairs": len(pairs),
        "positives": sum(p.should_hit for p in pairs),
        "target_precision": target_precision,
        "chosen_on": choose_on,
        "variants": variants,
    }
