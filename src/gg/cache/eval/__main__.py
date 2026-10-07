"""python -m gg.cache.eval: sweep the semantic threshold over the pair set, offline, no redis.

exit codes: 0 done, 2 harness or schema error.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

from gg.cache.config import EmbedderConfig
from gg.cache.embedders import build_embedder
from gg.cache.eval.pairs import load_pairs
from gg.cache.eval.sweep import sweep, thresholds
from gg.core.aio import CpuExecutor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gg.cache.eval", description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("evals/cache/pairs.jsonl"))
    parser.add_argument("--embedder", choices=("hashing", "fastembed"), default="hashing")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--target-precision", type=float, default=0.98)
    parser.add_argument("--max-tau", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--out", type=Path, default=None, help="write the result json here")
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    dim = args.dim or (384 if args.embedder == "fastembed" else 256)
    cpu = CpuExecutor(workers=1, queue_max=8)
    try:
        embedder = build_embedder(EmbedderConfig(provider=args.embedder, name=args.model, dim=dim), cpu)
        pairs = load_pairs(args.pairs)
        taus = thresholds(stop=args.max_tau, step=args.step)
        return await sweep(pairs, embedder, taus=taus, target_precision=args.target_precision)
    finally:
        cpu.shutdown()


def _summary(result: dict[str, Any]) -> str:
    embedder = result["embedder"]
    lines = [f"cache sweep  embedder={embedder['name']}:{embedder['dim']}  pairs={result['pairs']}"]
    for name, variant in result["variants"].items():
        at = variant["held_out_at_tau_star"]
        if at is None:
            lines.append(f"  {name:<18} no τ reaches precision {result['target_precision']}")
            continue
        lines.append(
            f"  {name:<18} τ*={variant['tau_star']:.3f}  held-out precision={at['precision'] or 0:.3f}  "
            f"hit_rate={at['hit_rate'] or 0:.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_summary(result))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
