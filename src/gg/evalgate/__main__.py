"""python -m gg.evalgate: run the guardrail and cache evals in replay mode and gate them (C11 §3.5).

exit codes (C11 §3.3): 0 pass, 1 gate failed, 2 harness, schema or git error.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson
from pydantic import ValidationError

from gg.config.loader import ConfigError
from gg.core.jsonutil import loads
from gg.evalgate.gates import AcceptedChange, added_entries, evaluate, parse_accepted, parse_suite
from gg.evalgate.refs import GitError, read_at_ref
from gg.evalgate.report import overall, render
from gg.evalgate.suites import BASELINES, RUNNERS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gg.evalgate", description=__doc__)
    parser.add_argument("--suite", action="append", choices=sorted(RUNNERS), help="repeatable; default all")
    parser.add_argument("--evals", type=Path, default=Path("evals"))
    parser.add_argument("--config", type=Path, default=Path("config"))
    parser.add_argument(
        "--base-ref",
        default=None,
        help="git ref to read baselines from; only accepted_changes entries added since it count",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out/eval"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or None,
        help="append the markdown summary here (default $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument("--update-baselines", action="store_true", help="rewrite baseline.json from this run")
    return parser


def _read(path: Path, base_ref: str | None) -> str | None:
    """file at the base ref when one is given, else the working tree; None when missing"""
    if base_ref is not None:
        if path.is_absolute():
            raise ValueError(f"{path}: paths must be relative to the repo root with --base-ref")
        data = read_at_ref(base_ref, path)
        return None if data is None else data.decode("utf-8")
    return path.read_text() if path.is_file() else None


def _baseline(path: Path, base_ref: str | None) -> dict[str, Any] | None:
    # a suite added after the base ref has no baseline there yet; fall back to the committed one
    text = _read(path, base_ref) or _read(path, None)
    return None if text is None else loads(text)


def _accepted(path: Path, base_ref: str | None) -> list[AcceptedChange]:
    head = parse_accepted(_read(path, None))
    if base_ref is None:
        return head
    return added_entries(head, parse_accepted(_read(path, base_ref)))


async def _run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for suite in args.suite or RUNNERS:
        suite_dir: Path = args.evals / suite
        spec = parse_suite((suite_dir / "suite.yaml").read_text())
        if spec.suite != suite:
            raise ValueError(f"{suite_dir / 'suite.yaml'} declares suite {spec.suite!r}")
        baseline = None if args.update_baselines else _baseline(suite_dir / "baseline.json", args.base_ref)
        raw = await RUNNERS[suite](spec, args.evals, args.config, baseline)
        if args.update_baselines:
            results[suite] = raw
            continue
        results[suite] = evaluate(
            raw, spec, baseline, _accepted(suite_dir / "accepted_changes.yaml", args.base_ref)
        )
    return results


def _write_baselines(args: argparse.Namespace, results: dict[str, dict[str, Any]]) -> None:
    for suite, result in results.items():
        path: Path = args.evals / suite / "baseline.json"
        # indented so baseline changes review as readable diffs
        path.write_bytes(orjson.dumps(BASELINES[suite](result), option=orjson.OPT_INDENT_2) + b"\n")
        print(f"wrote {path}")


def _write_outputs(args: argparse.Namespace, results: dict[str, dict[str, Any]]) -> str:
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for suite, result in results.items():
        (out / f"{suite}.json").write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    gate = {
        "status": overall(results),
        "base_ref": args.base_ref,
        "suites": {
            suite: {k: r.get(k) for k in ("status", "gates", "regressions", "accepted_regressions")}
            for suite, r in results.items()
        },
    }
    (out / "gate.json").write_bytes(orjson.dumps(gate, option=orjson.OPT_INDENT_2) + b"\n")
    summary = render(results, base_ref=args.base_ref)
    (out / "summary.md").write_text(summary)
    if args.summary is not None:
        with Path(args.summary).open("a") as f:
            f.write(summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results = asyncio.run(_run(args))
    except (ConfigError, ValidationError, ValueError, OSError, GitError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.update_baselines:
        _write_baselines(args, results)
        return 0
    print(_write_outputs(args, results))
    return 1 if overall(results) == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
