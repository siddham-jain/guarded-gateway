"""python -m gg.cli.routing_eval: the routing eval harness (C5 §10, C11 §3.8).

scores every prompt with the router scorer, answers it with both the weak and the strong model through
gg's own gateway (in process, so adapters, cost accounting and provider spend caps apply), grades the pairs
and sweeps alpha offline. every call is cached on disk; reruns only pay for what is missing.

exit codes: 0 done, 2 harness or config error, 3 refused or stopped by the spend cap.
"""

import argparse
import asyncio
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import orjson
import yaml
from fastapi import FastAPI
from pydantic import ValidationError

from gg.app.factory import Overrides, build_app
from gg.auth.keys import generate_key
from gg.config.loader import ConfigError, load_file
from gg.config.settings import Settings
from gg.core.clock import SystemClock
from gg.limits.spend_guard import InMemorySpendGuard, SpendGuardSettings
from gg.providers.base import ModelCatalog
from gg.routing.base import RoutingRequest, RoutingScore, RoutingScorer
from gg.routing.config import RoutingConfig
from gg.routing.eval.budget import Price, SpendCap
from gg.routing.eval.dryrun import mock_judge_params, mock_params
from gg.routing.eval.items import EvalItem, PairConfig, items_digest, limit_items, load_items, load_suite
from gg.routing.eval.judge import JudgePrompt
from gg.routing.eval.report import build_result, render_csv, render_markdown, render_svg
from gg.routing.eval.run import Generation, GenerationError, RoutingEvalRun, Stores
from gg.routing.eval.scorers import FakeScorer, build_jev_scorer, jev_cost_estimate_usd

PHASES = ("score", "generate", "judge")
EVAL_KEY_ID = "routing-eval"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gg.cli.routing_eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", type=Path, default=Path("evals/routing/suite.yaml"))
    parser.add_argument("--pair", default=None, help="model pair from the suite (default: ci with --dry-run)")
    parser.add_argument(
        "--dry-run", action="store_true", help="mock provider + fake scorer, end to end, no network"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="store + report dir (default evals/routing/runs/<pair>)"
    )
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument(
        "--max-usd", type=float, default=1.0, help="hard cap on billed spend for this invocation"
    )
    parser.add_argument("--limit", type=int, default=None, help="first N items, round-robin over categories")
    parser.add_argument("--phases", default=",".join(PHASES), help="comma list of score,generate,judge")
    parser.add_argument("--offline", action="store_true", help="no calls: report from the stores only")
    parser.add_argument("--estimate-only", action="store_true", help="print the cost estimate and exit")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=None, help="alpha to report at (default routing.yaml)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for APGR CIs")
    return parser


def _keys_file(directory: Path, token_hash: str, prefix: str, models: Sequence[str]) -> Path:
    entry = {
        "id": EVAL_KEY_ID,
        "name": "routing eval (ephemeral)",
        "hash": token_hash,
        "prefix": prefix,
        "created_at": datetime.now(UTC).date().isoformat(),
        "allowed_models": sorted(set(models)),
        "rate_limits": {"rpm": 6000, "tpm": 50_000_000, "max_concurrent": 64},
        "cache": {"scope": "off"},
        "limits": {"max_completion_tokens": 16_000},
    }
    path = directory / "keys.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "keys": [entry]}, sort_keys=False))
    return path


def _settings(args: argparse.Namespace, pair: PairConfig, keys_file: Path) -> Settings:
    common: dict[str, Any] = {
        "config_dir": args.config_dir,
        "keys_file": keys_file,
        "model_profile": pair.profile,
        "log_level": "warning",
        "log_format": "console",
        "redis_url": None,
        "langfuse": {"enabled": False},
    }
    if args.dry_run:
        return Settings(_env_file=None, env="test", jev_api_key=None, **common)  # pyright: ignore[reportCallIssue]
    return Settings(**common)


def _prices(catalog: ModelCatalog, pair: PairConfig) -> dict[str, Price]:
    now = datetime.now(UTC)
    prices: dict[str, Price] = {}
    for role in (pair.weak, pair.strong, pair.judge):
        try:
            deployment = catalog.get(role.model)
        except KeyError:
            raise ValueError(
                f"deployment {role.model} is not in the catalog for profile {pair.profile}; "
                "add it to config/models.yaml (see evals/routing/README.md)"
            ) from None
        pricing = deployment.pricing.at(now)
        if pricing is None:
            raise ValueError(f"deployment {role.model} has no price at {now.date()}")
        prices[role.model] = Price(
            float(pricing.input), float(pricing.output), billed=deployment.pricing.billed
        )
    return prices


class GatewayGenerator:
    """posts to /v1/chat/completions of the in-process app with fallback and caching off"""

    def __init__(self, client: httpx2.AsyncClient, token: str, prices: Mapping[str, Price]) -> None:
        self._client = client
        self._headers = {"authorization": f"Bearer {token}"}
        self._prices = prices

    async def generate(
        self, model: str, messages: Sequence[Mapping[str, str]], params: Mapping[str, Any]
    ) -> Generation:
        body = {
            "model": model,
            "messages": list(messages),
            **params,
            "gg": {"fallback": False, "cache": "off", "semantic_cache": False},
        }
        started = time.perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=body, headers=self._headers)
        except httpx2.HTTPError as exc:
            raise GenerationError(f"{model}: {type(exc).__name__}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        if resp.status_code != 200:
            raise GenerationError(f"{model}: http {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        price = self._prices[model]
        cost = price.cost(prompt_tokens, completion_tokens)
        return Generation(
            text=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason"),
            served_model=resp.headers.get("x-gg-model", model),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            reasoning_tokens=int(details.get("reasoning_tokens", 0)),
            cost_usd=cost,
            billed_usd=cost if price.billed else 0.0,
            latency_ms=latency_ms,
        )


class CachedScoresOnly:
    """when the score phase is skipped: carries the cached scorer version so stored scores are found"""

    name = "cached"

    def __init__(self, version: str) -> None:
        self.version = version

    async def score(self, req: RoutingRequest, /) -> RoutingScore:
        raise GenerationError("the score phase is not enabled")


def _scorer(
    args: argparse.Namespace,
    phases: Sequence[str],
    items: Sequence[EvalItem],
    settings: Settings,
    stores: Stores,
    http: httpx2.AsyncClient,
) -> tuple[RoutingScorer, float]:
    if args.dry_run:
        return FakeScorer({f"eval-{i.id}": i.expected_tier for i in items}), 0.0
    if "score" in phases:
        if settings.jev_api_key is None:
            raise ValueError("GG_JEV_API_KEY is not set; the score phase needs it (or drop it from --phases)")
        scorer = build_jev_scorer(
            args.config_dir, settings.jev_api_key.get_secret_value(), http, SystemClock()
        )
        return scorer, jev_cost_estimate_usd(args.config_dir)
    versions = sorted({str(r.get("scorer_version")) for r in stores.scores})
    if len(versions) > 1:
        raise ValueError(
            f"score store holds several scorer versions ({', '.join(versions)}); run the score phase"
        )
    return CachedScoresOnly(versions[0] if versions else "none"), 0.0


def _write_reports(out: Path, result: dict[str, Any]) -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    markdown = render_markdown(result)
    (out / "report.md").write_text(markdown)
    (out / "curves.csv").write_text(render_csv(result))
    (out / "cost_quality.svg").write_text(render_svg(result))
    return markdown


async def _run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    pair_name = args.pair or ("ci" if args.dry_run else None)
    if pair_name is None:
        raise ValueError("pass --pair (one of: " + ", ".join(suite.pairs) + ") or --dry-run")
    if pair_name not in suite.pairs:
        raise ValueError(f"unknown pair {pair_name}; known: {', '.join(suite.pairs)}")
    if args.dry_run and suite.pairs[pair_name].profile != "ci":
        raise ValueError("--dry-run only runs a pair on the ci (mock) profile")
    pair = suite.pairs[pair_name]
    all_items = load_items(suite.items)
    items = limit_items(all_items, args.limit)
    phases = [] if args.offline else [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = set(phases) - set(PHASES)
    if unknown:
        raise ValueError(f"unknown phases: {', '.join(sorted(unknown))}")
    out: Path = args.out or Path("evals/routing/runs") / ("dry-run" if args.dry_run else pair_name)
    routing = load_file(args.config_dir / "routing.yaml", RoutingConfig)
    alpha = routing.policy.threshold if args.alpha is None else args.alpha
    judge_prompt = JudgePrompt.load(suite.judge_prompt)

    key = generate_key("live")
    async with AsyncExitStack() as stack:
        tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="gg-routing-eval-")))
        keys_file = _keys_file(
            tmp, key.hash, key.prefix, [pair.weak.model, pair.strong.model, pair.judge.model]
        )
        settings = _settings(args, pair, keys_file)
        overrides = (
            Overrides(
                spend_guard=InMemorySpendGuard(
                    SpendGuardSettings(_env_file=None),  # pyright: ignore[reportCallIssue]
                    clock=SystemClock(),
                )
            )
            if args.dry_run
            else None
        )
        app: FastAPI = build_app(settings, overrides=overrides)
        prices = _prices(app.state.services.catalog, pair)
        jev_http = await stack.enter_async_context(httpx2.AsyncClient())
        stores = Stores.open(out)
        scorer, scorer_cost = _scorer(args, phases, items, settings, stores, jev_http)
        cap = SpendCap(args.max_usd)
        gateway = await stack.enter_async_context(
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="http://gg.eval", timeout=600
            )
        )
        runner = RoutingEvalRun(
            items,
            pair,
            stores=stores,
            generator=GatewayGenerator(gateway, key.token, prices),
            scorer=scorer,
            prices=prices,
            judge_prompt=judge_prompt,
            cap=cap,
            scorer_cost_usd=scorer_cost,
            concurrency=args.concurrency,
            extra_params=mock_params if args.dry_run else None,
            judge_extra=mock_judge_params if args.dry_run else None,
        )
        estimate = runner.estimate(phases)
        print(f"routing eval  pair={pair_name}  items={len(items)}/{len(all_items)}  store={out}")
        print("\n".join(estimate.lines()))
        if args.estimate_only:
            return 0
        if estimate.billed_expected_usd > args.max_usd:
            print(
                f"refusing: expected billed spend ${estimate.billed_expected_usd:.4f} is above --max-usd "
                f"${args.max_usd:.2f}",
                file=sys.stderr,
            )
            return 3
        if phases:
            await stack.enter_async_context(app.router.lifespan_context(app))
            await runner.run(phases)
    collected = runner.collect()
    meta = {
        "mode": "offline" if args.offline else ("dry-run" if args.dry_run else "live"),
        "pair": {
            "name": pair_name,
            "profile": pair.profile,
            "weak": pair.weak.model,
            "strong": pair.strong.model,
            "judge": pair.judge.model,
        },
        "scorer_version": runner.scorer.version,
        "judge_prompt": judge_prompt.version,
        "suite_version": suite.suite_version,
        "dataset": {"sha256": items_digest(all_items), "n_items": len(items)},
        "cost": {
            "spent_usd": round(cap.spent_usd, 6),
            "cap_usd": args.max_usd,
            "budget_stop": runner.budget_stop,
            "phases": {name: stats.to_json() for name, stats in runner.stats.items()},
        },
    }
    result = build_result(collected, items, meta=meta, current_alpha=alpha, resamples=args.bootstrap)
    markdown = _write_reports(out, result)
    print(markdown)
    for name, stats in runner.stats.items():
        for error in stats.errors[:5]:
            print(f"  {name} error: {error}", file=sys.stderr)
    return 3 if runner.budget_stop else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (ConfigError, ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
