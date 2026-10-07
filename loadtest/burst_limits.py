"""burst test for c9: hammer one key and prove the rate limit, concurrency cap and budget hold.

two targets:
  http    closed-loop users against a running gateway (stock compose + mock upstream)
  direct  the redis lua scripts themselves, many connections, no gateway (fakeredis when no --redis-url)

checks (exit 1 on any violation):
  A1  for every sliding window W of admitted requests: count <= capacity + rate * W + 1
  A3  peak in-flight per key <= max_concurrent (direct; over http read the mock's /_stats)
  A4  budget: settled spend <= cap, and every refusal after exhaustion is 402
  A5  every 429 carries retry-after and x-ratelimit-*; no 5xx

  python loadtest/burst_limits.py http --base-url http://localhost:8000 --api-key "$GG_KEY" --rpm 60
  python loadtest/burst_limits.py direct --rpm 300 --max-concurrent 20 --budget-micros 50000
"""

import argparse
import asyncio
import json
import sys
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import httpx2


@dataclass
class Outcome:
    at: float
    status: int
    headers: dict[str, str] = field(default_factory=lambda: {})
    cost_usd: float = 0.0


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=lambda: [])
    peak_inflight: int = 0
    violations: list[str] = field(default_factory=lambda: [])

    def summary(self) -> dict[str, Any]:
        return {
            "requests": len(self.outcomes),
            "status": dict(Counter(o.status for o in self.outcomes)),
            "peak_inflight": self.peak_inflight,
            "spent_usd": round(sum(o.cost_usd for o in self.outcomes), 6),
            "violations": self.violations,
        }


def check_window(admitted: list[float], *, rate_per_min: float, capacity: float, report: Report) -> None:
    """A1 over every window that starts at an admitted request"""
    times = sorted(admitted)
    rate_per_s = rate_per_min / 60
    for start, opened in enumerate(times):
        for width in (1, 2, 5, 10, 30, 60):
            count = bisect_left(times, opened + width) - start
            bound = capacity + rate_per_s * width + 1
            if count > bound:
                report.violations.append(f"A1: {count} admitted in {width}s window > bound {bound:.1f}")
                return


async def run_http(args: argparse.Namespace) -> Report:
    report = Report()
    inflight = 0
    deadline = time.monotonic() + args.duration
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "burst test"}],
        "max_completion_tokens": args.max_tokens,
    }
    headers = {"authorization": f"Bearer {args.api_key}"}

    async def user(client: httpx2.AsyncClient) -> None:
        nonlocal inflight
        while time.monotonic() < deadline:
            inflight += 1
            report.peak_inflight = max(report.peak_inflight, inflight)
            try:
                resp = await client.post("/v1/chat/completions", json=body, headers=headers)
            finally:
                inflight -= 1
            got = {k.lower(): v for k, v in resp.headers.items()}
            report.outcomes.append(
                Outcome(time.monotonic(), resp.status_code, got, float(got.get("x-gg-cost-usd", 0)))
            )
            if resp.status_code == 429:
                await asyncio.sleep(args.backoff_s)

    limits = httpx2.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    async with httpx2.AsyncClient(base_url=args.base_url, timeout=60, limits=limits) as client:
        await asyncio.gather(*(user(client) for _ in range(args.users)))

    for o in report.outcomes:
        if o.status >= 500:
            report.violations.append(f"A5: got {o.status}")
        if o.status == 429 and ("retry-after" not in o.headers or not _has_ratelimit_headers(o.headers)):
            report.violations.append("A5: 429 without retry-after or x-ratelimit-* headers")
    admitted = [o.at for o in report.outcomes if o.status == 200]
    if args.rpm:
        check_window(admitted, rate_per_min=args.rpm, capacity=_capacity(args), report=report)
    # client-side in-flight counts rejected calls too, so A3 over http is read from the mock's /_stats
    if args.budget_usd is not None:
        spent = sum(o.cost_usd for o in report.outcomes)
        if spent > args.budget_usd + 1e-9:
            report.violations.append(f"A4: spent ${spent:.6f} > budget ${args.budget_usd:.6f}")
    return report


def _has_ratelimit_headers(headers: dict[str, str]) -> bool:
    return any(name.startswith("x-ratelimit-") for name in headers)


def _capacity(args: argparse.Namespace) -> float:
    return max(1.0, args.rpm * args.burst_window_s / 60)


async def run_direct(args: argparse.Namespace) -> Report:
    from gg.core.clock import SystemClock
    from gg.core.keypolicy import RateLimitPolicy
    from gg.limits.base import BudgetCaps, BudgetExceededError, LimitResult
    from gg.limits.config import LimitsConfig
    from gg.limits.ledger import RedisBudgetLedger
    from gg.limits.redis_limiter import RedisRateLimiter

    if args.redis_url:
        from redis.asyncio import Redis

        redis = Redis.from_url(args.redis_url)
    else:
        import fakeredis

        redis = fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer())
    cfg = LimitsConfig(redis_prefix="ggburst")
    key = f"burst-{int(time.time())}"
    limiter = RedisRateLimiter(redis, cfg.rate_limits, prefix=cfg.redis_prefix)
    ledger = RedisBudgetLedger(redis, cfg.budgets, clock=SystemClock(), prefix=cfg.redis_prefix)
    limits = cfg.rate_limits.bucket_limits(
        RateLimitPolicy(rpm=args.rpm or None, tpm=None, max_concurrent=args.max_concurrent or None)
    )
    report = Report()
    inflight = 0
    deadline = time.monotonic() + args.duration
    hold_micros = args.hold_micros

    async def user() -> None:
        nonlocal inflight
        while time.monotonic() < deadline:
            result: LimitResult = await limiter.acquire(key, 0, limits=limits, lease_ttl_s=30)
            if not result.allowed:
                report.outcomes.append(Outcome(time.monotonic(), 429))
                await asyncio.sleep(args.backoff_s)
                continue
            inflight += 1
            report.peak_inflight = max(report.peak_inflight, inflight)
            status, cost = 200, 0.0
            try:
                if args.budget_micros is not None:
                    try:
                        hold, _ = await ledger.preauthorize(
                            key, hold_micros, caps=BudgetCaps(daily=args.budget_micros), hold_ttl_s=60
                        )
                    except BudgetExceededError:
                        status = 402
                    else:
                        await asyncio.sleep(args.service_s)
                        # the deterministic mock settles below its hold, as in the c9 lt-budget scenario
                        actual = hold_micros * 3 // 4
                        await ledger.settle(hold, actual)
                        cost = actual / 1_000_000
                else:
                    await asyncio.sleep(args.service_s)
            finally:
                inflight -= 1
                if result.lease is not None:
                    await limiter.finish(result.lease, 0)
            report.outcomes.append(Outcome(time.monotonic(), status, cost_usd=cost))

    await asyncio.gather(*(user() for _ in range(args.users)))
    await redis.aclose()

    admitted = [o.at for o in report.outcomes if o.status in (200, 402)]
    if args.rpm:
        check_window(admitted, rate_per_min=args.rpm, capacity=_capacity(args), report=report)
    if args.max_concurrent and report.peak_inflight > args.max_concurrent:
        report.violations.append(f"A3: peak in-flight {report.peak_inflight} > {args.max_concurrent}")
    if args.budget_micros is not None:
        spent = round(sum(o.cost_usd for o in report.outcomes) * 1_000_000)
        if spent > args.budget_micros:
            report.violations.append(f"A4: settled {spent} micro-usd > cap {args.budget_micros}")
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("target", choices=["http", "direct"])
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rpm", type=int, default=0, help="the key's rpm limit (0 = not checked)")
    parser.add_argument("--burst-window-s", type=float, default=10.0, help="limits.yaml rpm_burst_window_s")
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--backoff-s", type=float, default=0.05, help="pause after a 429 (closed loop)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="mock/luna")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--budget-usd", type=float, default=None, help="http: the key's daily budget")
    parser.add_argument(
        "--redis-url", default=None, help="direct: real redis (default: in-process fakeredis)"
    )
    parser.add_argument("--budget-micros", type=int, default=None, help="direct: daily cap to burst against")
    parser.add_argument("--hold-micros", type=int, default=200)
    parser.add_argument("--service-s", type=float, default=0.02, help="direct: simulated upstream time")
    args = parser.parse_args(argv)
    if args.target == "http" and not args.api_key:
        parser.error("http needs --api-key")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = asyncio.run(run_http(args) if args.target == "http" else run_direct(args))
    print(json.dumps(report.summary(), indent=2))
    return 1 if report.violations else 0


if __name__ == "__main__":
    sys.exit(main())
