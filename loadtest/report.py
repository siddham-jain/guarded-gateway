"""summaries and the markdown report, computed from raw samples and gateway request logs only"""

import csv
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

QUANTILES = (50, 95, 99)


def percentile(values: Sequence[float], q: float) -> float | None:
    """linear interpolation between closest ranks (numpy's default)"""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {f"p{q}": percentile(values, q) for q in QUANTILES}
    out["mean"] = sum(values) / len(values) if values else None
    return out


def load_samples(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("samples-*.csv")):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _floats(rows: Iterable[Mapping[str, str]], field: str) -> list[float]:
    return [float(r[field]) for r in rows if r.get(field)]


def _rate(rows: Sequence[Mapping[str, str]], fallback_s: float) -> float | None:
    """achieved rps over the span of recorded request starts (locust's own start-up delay excluded)"""
    starts = [float(r["started"]) for r in rows]
    span = max(starts) - min(starts) if len(starts) > 1 else 0.0
    window = span if span > fallback_s / 2 else fallback_s
    return len(rows) / window if window > 0 else None


def summarise_client(rows: Sequence[Mapping[str, str]], measured_s: float) -> dict[str, Any]:
    ok = [r for r in rows if r["ok"] == "1"]
    caches: dict[str, int] = {}
    for r in rows:
        if r.get("cache"):
            caches[r["cache"]] = caches.get(r["cache"], 0) + 1
    return {
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "error_rate": (len(rows) - len(ok)) / len(rows) if rows else None,
        "rps": _rate(rows, measured_s),
        "total_ms": quantiles(_floats(ok, "total_ms")),
        "ttft_ms": quantiles(_floats(ok, "ttft_ms")),
        "gw_header_ms": quantiles(_floats(ok, "gw_ms")),
        "cache": caches,
    }


def read_request_logs(path: Path, start: float, end: float) -> list[dict[str, Any]]:
    """request.completed records whose timestamp falls in [start, end] (epoch seconds)"""
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("rb") as f:
        for line in f:
            if b'"request.completed"' not in line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            at = datetime.fromisoformat(record["timestamp"]).timestamp()
            if start <= at <= end:
                out.append(record)
    return out


def summarise_server(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    timings = [r.get("timing_ms") or {} for r in records if r.get("status") == 200]

    def pick(name: str) -> list[float]:
        return [float(t[name]) for t in timings if t.get(name) is not None]

    stages: dict[str, list[float]] = {}
    for t in timings:
        for name, ms in (t.get("stages") or {}).items():
            if name != "terminal":
                stages.setdefault(name, []).append(float(ms))
    return {
        "requests": len(records),
        "overhead_ms": quantiles(pick("overhead")),
        "pre_upstream_ms": quantiles(pick("pre_upstream")),
        "ttft_added_ms": quantiles(pick("ttft_added")),
        "stream_tail_ms": quantiles(pick("stream_tail")),
        "stages_p50_ms": {name: percentile(v, 50) for name, v in sorted(stages.items())},
    }


def fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _q(block: Mapping[str, Any], key: str, q: str, digits: int = 1) -> str:
    return fmt((block.get(key) or {}).get(q), digits)


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def find(runs: Sequence[Mapping[str, Any]], **match: Any) -> Mapping[str, Any] | None:
    return next((r for r in runs if all(r.get(k) == v for k, v in match.items())), None)


def overhead_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """client-observed delta vs direct-to-mock at the same offered rate, next to the server's own numbers"""
    rows: list[list[str]] = []
    for run in runs:
        if run["cell"] == "direct" or run.get("kind") != "fixed" or run["scenario"] == "cachehit":
            continue
        base = find(runs, cell="direct", scenario=run["scenario"], kind="fixed", target_rps=run["target_rps"])
        if base is None:
            continue
        field = "ttft_ms" if run["scenario"] == "stream" else "total_ms"
        server = "ttft_added_ms" if run["scenario"] == "stream" else "overhead_ms"
        metric = run.get("metric_overhead_mean_ms") or {}
        client_delta = [
            _sub(run["client"][field][q], base["client"][field][q]) for q in ("p50", "p95", "p99")
        ]
        rows.append(
            [
                run["cell"],
                run["scenario"],
                f"{run['target_rps']:g}",
                " / ".join(fmt(v) for v in client_delta),
                " / ".join(_q(run["server"], server, q) for q in ("p50", "p95", "p99")),
                " / ".join(_q(run["server"], "overhead_ms", q) for q in ("p50", "p99"))
                if run["scenario"] == "stream"
                else "-",
                " / ".join(_q(run["client"], "gw_header_ms", q) for q in ("p50", "p99")),
                fmt(metric.get(f"{'true' if run['scenario'] == 'stream' else 'false'}/total"), 2),
            ]
        )
    return rows


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def run_rows(runs: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run in runs:
        c, s, res = run["client"], run.get("server") or {}, run.get("resources") or {}
        rows.append(
            [
                run["cell"],
                run["scenario"],
                f"{run['target_rps']:g}",
                fmt(c["rps"]),
                f"{c['requests']}",
                f"{100 * (c['error_rate'] or 0):.2f}%",
                " / ".join(_q(c, "total_ms", q) for q in ("p50", "p95", "p99")),
                " / ".join(_q(c, "ttft_ms", q) for q in ("p50", "p99"))
                if run["scenario"] == "stream"
                else "-",
                " / ".join(_q(s, "overhead_ms", q, 2) for q in ("p50", "p99")) if s else "-",
                fmt(res.get("gateway_cpu_mean")),
                fmt(res.get("gateway_rss_max_mb"), 0),
                fmt(res.get("locust_proc_cpu_max")),
            ]
        )
    return rows


def knee(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the lowest step's p99"""
    if not steps:
        return None
    floor = steps[0]["client"]["total_ms"]["p99"] or 0
    best = None
    for step in steps:
        c = step["client"]
        if (
            (c["rps"] or 0) >= 0.95 * step["target_rps"]
            and (c["error_rate"] or 0) < 0.01
            and (c["total_ms"]["p99"] or math.inf) <= 2 * max(floor, 1)
        ):
            best = step
    return best


def render(meta: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], extras: Mapping[str, Any]) -> str:
    env = meta["env"]
    out = [
        f"# GG load test - {meta['started']}",
        "",
        f"Host: {env['cpu']}, {env['cores']} cores, {env['memory_gb']} GB RAM, {env['os']}, "
        f"Python {env['python']}, locust {env['locust']}. Local processes, no containers, no CPU pinning; "
        "the load generator shares the host.",
        "",
        "Conditions: gateway `gg serve`, 1 uvicorn worker (uvloop, httptools), model profile `loadtest`, "
        "Langfuse export off, ML guards off unless the cell says so. "
        f"Mock upstream, 1 worker: {meta['mock']}. "
        f"Fixed-rate runs: {meta['warmup_s']} s warmup (discarded) + {meta['duration_s']} s measured, "
        "open-ish arrivals (constant_throughput users with random start phase). "
        "Percentiles come from raw per-request samples. "
        f"First request after start (ms): {meta.get('cold_start_ms') or '-'}.",
        "",
        "Cells: " + "; ".join(f"`{k}` {v}" for k, v in meta["cells"].items()),
        "",
        "## Gateway overhead",
        "",
        "Client delta: gateway percentile minus direct-to-mock percentile at the same offered rate "
        "(non-stream: total latency; stream: time to first content chunk). Server: the gateway's own "
        "per-request `timing_ms` from the `request.completed` log (non-stream `overhead` = total - upstream; "
        "stream `ttft_added`, and `overhead` = ttft_added + stream_tail). Server-Timing `gw` is the response "
        "header as the client saw it; the metric column is the mean of `gg_gateway_overhead_seconds` "
        "(phase total) over the run.",
        "",
        table(
            [
                "cell",
                "scenario",
                "RPS",
                "client delta p50/p95/p99 ms",
                "server p50/p95/p99 ms",
                "stream overhead p50/p99 ms",
                "Server-Timing gw p50/p99 ms",
                "gg_gateway_overhead_seconds mean ms",
            ],
            overhead_rows(runs),
        ),
        "",
        "## All runs",
        "",
        table(
            [
                "cell",
                "scenario",
                "offered RPS",
                "achieved RPS",
                "requests",
                "errors",
                "latency p50/p95/p99 ms",
                "TTFT p50/p99 ms",
                "server overhead p50/p99 ms",
                "gw CPU % mean",
                "gw RSS MB",
                "locust CPU % max (1 proc)",
            ],
            run_rows(runs),
        ),
    ]
    stage_runs = [
        r for r in runs if r.get("server") and r["scenario"] == "nonstream" and r.get("kind") == "fixed"
    ]
    if stage_runs:
        names = sorted({n for r in stage_runs for n in r["server"]["stages_p50_ms"]})
        out += [
            "",
            "## Stage time p50 (ms, non-stream, from the request log)",
            "",
            table(
                ["stage", *[f"{r['cell']} @{r['target_rps']:g}" for r in stage_runs]],
                [[n, *[fmt(r["server"]["stages_p50_ms"].get(n), 3) for r in stage_runs]] for n in names],
            ),
        ]
    for cell, steps in (extras.get("knee") or {}).items():
        best = knee(steps)
        out += [
            "",
            f"## Throughput steps - `{cell}`",
            "",
            "Knee = highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the first "
            "step's p99.",
            "",
            table(
                [
                    "offered RPS",
                    "achieved RPS",
                    "errors",
                    "p50/p95/p99 ms",
                    "gw CPU % mean",
                    "locust CPU % max (1 proc)",
                ],
                [
                    [
                        f"{s['target_rps']:g}",
                        fmt(s["client"]["rps"]),
                        f"{100 * (s['client']['error_rate'] or 0):.2f}%",
                        " / ".join(_q(s["client"], "total_ms", q) for q in ("p50", "p95", "p99")),
                        fmt((s.get("resources") or {}).get("gateway_cpu_mean")),
                        fmt((s.get("resources") or {}).get("locust_proc_cpu_max")),
                    ]
                    for s in steps
                ],
            ),
            "",
            f"Knee: **{fmt(best['client']['rps'], 0) if best else 'below first step'} RPS**"
            + (f" (offered {best['target_rps']:g})" if best else ""),
        ]
    if burst := extras.get("burst"):
        out += [
            "",
            "## Burst against a limited key",
            "",
            "```json",
            orjson.dumps(burst, option=orjson.OPT_INDENT_2).decode(),
            "```",
        ]
    mock = {
        f"{r['cell']} {r['scenario']} @{r['target_rps']:g}": r["upstream_calls"]
        for r in runs
        if r["cell"] != "direct" and r.get("kind") == "fixed"
    }
    if mock:
        out += [
            "",
            "## Mock upstream calls per run",
            "",
            table(["run", "upstream calls"], [[k, str(v)] for k, v in mock.items()]),
        ]
    if profile := extras.get("profile"):
        out += ["", "## Profile (cProfile, main thread, non-stream run)", "", profile]
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """re-renders report.md of a bundle from its summary.json"""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m loadtest.report <bundle dir>", file=sys.stderr)
        return 2
    bundle = Path(args[0])
    data = orjson.loads((bundle / "summary.json").read_bytes())
    (bundle / "report.md").write_text(render(data["meta"], data["runs"], data["extras"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
