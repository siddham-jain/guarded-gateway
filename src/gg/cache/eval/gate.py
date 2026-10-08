"""pair outcomes at one threshold for the ci gate (C11 §3.9): result json, item regressions, baseline"""

import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from gg.cache.base import Embedder, SemanticVerifier
from gg.cache.eval.pairs import Pair
from gg.cache.eval.sweep import Scored, confusion, score_pairs, verify_pairs
from gg.core.jsonutil import canonical_json, sha256_hex

SUITE = "cache"
SUITE_VERSION = "1.0.0"

_RANK = {"hit": 1, "missed": 0, "rejected": 1, "false_hit": 0}


def outcome(scored: Scored, tau: float) -> str:
    hit = scored.distance is not None and scored.distance <= tau and scored.verified is not False
    if scored.pair.should_hit:
        return "hit" if hit else "missed"
    return "false_hit" if hit else "rejected"


def _rate(num: int, den: int, direction: str) -> dict[str, Any]:
    return {"value": round(num / den, 4) if den else None, "n": den, "k": num, "direction": direction}


def _metrics(scored: Sequence[Scored], tau: float, prefix: str) -> dict[str, Any]:
    c = confusion(scored, tau)
    return {
        f"{prefix}precision": _rate(c["tp"], c["tp"] + c["fp"], "higher"),
        f"{prefix}hit_rate": _rate(c["tp"], c["tp"] + c["fn"], "higher"),
    }


def regressions(items: Sequence[dict[str, Any]], baseline: dict[str, Any] | None) -> list[dict[str, str]]:
    """only no_hit pairs gate: a pair the cache rejected in the baseline now serves a wrong answer"""
    base_items: dict[str, str] = (baseline or {}).get("items", {})
    out: list[dict[str, str]] = []
    for item in items:
        before = base_items.get(item["id"])
        if before is None or item["should_hit"]:
            continue
        if _RANK[item["outcome"]] < _RANK.get(before, 0):
            out.append({"id": item["id"], "from": before, "to": item["outcome"]})
    return out


def pair_key(pair: Pair) -> str:
    return sha256_hex(f"{pair.anchor}\n{pair.candidate}")[:16]


async def run_gate(
    pairs: Sequence[Pair],
    embedder: Embedder,
    *,
    tau: float,
    baseline: dict[str, Any] | None,
    num_sig: bool = True,
    verifier: SemanticVerifier | None = None,
    recorded: Mapping[str, float | None] | None = None,
) -> dict[str, Any]:
    """`recorded` replays an earlier run's distances (by pair_key) instead of embedding; a new pair misses"""
    if recorded is None:
        scored = await score_pairs(pairs, embedder, num_sig=num_sig)
    else:
        scored = [Scored(p, recorded.get(pair_key(p))) for p in pairs]
    if verifier is not None:
        scored = await verify_pairs(scored, verifier, tau)
    test = [s for s in scored if s.pair.split == "test"]
    dev = [s for s in scored if s.pair.split == "dev"]
    items = [
        {
            "id": s.pair.id,
            "split": s.pair.split,
            "category": s.pair.category,
            "should_hit": s.pair.should_hit,
            "distance": None if s.distance is None else round(s.distance, 6),
            "verified": s.verified,
            "outcome": outcome(s, tau),
        }
        for s in scored
    ]
    diff = regressions(items, baseline)
    digest_input = canonical_json([p.model_dump(mode="json") for p in pairs])
    return {
        "schema_version": 1,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "run_id": datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "mode": "replay",
        "embedder": {"name": embedder.name, "dim": embedder.dim},
        "threshold": tau,
        "num_sig": num_sig,
        "dataset": {
            "items_sha256": "sha256:" + sha256_hex(digest_input),
            "n_items": len(pairs),
            "splits": {"dev": len(dev), "test": len(test)},
        },
        "env": {
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}".lower(),
        },
        "metrics": {**_metrics(test, tau, ""), **_metrics(dev, tau, "dev_")},
        "verifier_errors": sum(s.verify_failed for s in scored),
        "items": items,
        "regressions": diff,
        "gates": [],
        "status": "fail" if diff else "pass",
    }


def baseline_from(result: dict[str, Any]) -> dict[str, Any]:
    embedder = result["embedder"]
    return {
        "schema_version": 1,
        "suite": SUITE,
        "recorded_from": {
            "embedder": f"{embedder['name']}:{embedder['dim']}",
            "threshold": result["threshold"],
            "num_sig": result["num_sig"],
        },
        "items": {i["id"]: i["outcome"] for i in result["items"]},
        "metrics": {k: v["value"] for k, v in result["metrics"].items() if k in ("precision", "hit_rate")},
    }
