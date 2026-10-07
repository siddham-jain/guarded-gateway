"""python -m gg.guardrails.eval: score the guardrail eval items in process, no network.

exit codes (C11 §3.3): 0 pass, 1 item regression against the baseline, 2 harness or schema error.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from gg.config.loader import ConfigError
from gg.core.clock import SystemClock
from gg.core.jsonutil import dumps, loads
from gg.core.schema import ChatRequest
from gg.guardrails.eval.items import load_items
from gg.guardrails.eval.runner import GuardrailEval, baseline_from, eval_key, report
from gg.guardrails.setup import build_guardrails


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gg.guardrails.eval", description=__doc__)
    parser.add_argument("--items", type=Path, default=Path("evals/guardrails/items"))
    parser.add_argument("--policies", type=Path, default=Path("config/policies"))
    parser.add_argument("--policy-id", default="default")
    parser.add_argument("--baseline", type=Path, default=Path("evals/guardrails/baseline.json"))
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    parser.add_argument("--out", type=Path, default=None, help="write the result json here")
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline from this run")
    return parser


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    guardrails = build_guardrails(args.policies, clock=SystemClock(), keys=[eval_key(args.policy_id)])
    items = [i for i in load_items(args.items) if args.split in ("all", i.split)]
    runner = GuardrailEval(guardrails.policies, guardrails.engine, policy_id=args.policy_id)
    results = [await runner.run_item(item) for item in items]
    baseline = loads(args.baseline.read_bytes()) if args.baseline.is_file() else None
    probe = ChatRequest.model_validate({"model": "gg/auto", "messages": [{"role": "user", "content": "x"}]})
    return report(results, items, runner.policy(probe), baseline)


def _line(name: str, metric: dict[str, Any]) -> str:
    if "k" not in metric:
        return f"  {name:<22} {metric['value']}"
    low, high = metric["ci95"]
    return f"  {name:<22} {metric['k']}/{metric['n']} = {metric['value']:.3f}  [{low:.3f}, {high:.3f}]"


def _summary(result: dict[str, Any]) -> str:
    policy = result["policy"]
    lines = [f"guardrails eval  policy={policy['id']}@{policy['version']}+{policy['hash']}"]
    lines += [_line(name, metric) for name, metric in result["metrics"].items()]
    lines += [f"  gate {g['name']}: {g['status']} ({g['detail']})" for g in result["gates"]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (ConfigError, ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_summary(result))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(dumps(result))
    if args.update_baseline:
        # indented so baseline changes review as readable diffs
        args.baseline.write_bytes(orjson.dumps(baseline_from(result), option=orjson.OPT_INDENT_2) + b"\n")
        return 0
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
