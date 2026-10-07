"""starts the mock upstream and the gateway, runs headless locust per cell and writes a result bundle.

  .venv/bin/python -m loadtest.run                 # standard run (~15 min)
  .venv/bin/python -m loadtest.run --smoke         # perf smoke for ci (~30 s), exit 1 on errors
  .venv/bin/python -m loadtest.run --cells bare full --skip-knee --skip-profile

the gateway runs with its cwd in the bundle directory, so the repo .env (provider keys, langfuse, promptguard)
is never loaded; only the GG_* variables set here reach it.
"""

import argparse
import contextlib
import gzip
import os
import platform
import pstats
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import orjson
import psutil
import yaml

from loadtest import report

ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = Path(sys.executable).parent
LOCUSTFILE = ROOT / "loadtest" / "locustfile.py"
KEYS_FILE = ROOT / "tests" / "fixtures" / "keys" / "loadtest.yaml"
TOKENS = yaml.safe_load((ROOT / "tests" / "fixtures" / "keys" / "tokens.yaml").read_text())
MOCK_PORT = 9100
GATEWAY_PORT = 8100
MOCK_URL = f"http://127.0.0.1:{MOCK_PORT}"
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}"
MOCK_ENV = {"MOCK_TTFT_MS": "50", "MOCK_ITL_MS": "5", "MOCK_OUTPUT_TOKENS": "40"}


@dataclass(frozen=True)
class Cell:
    name: str
    description: str
    guards: bool = True
    cache: bool = True
    redis: bool = False
    ml: bool = False
    first_window_chars: int | None = None


CELLS = {
    c.name: c
    for c in (
        Cell("bare", "guards off, cache off, no redis", guards=False, cache=False),
        Cell("guards", "rule-based input/output guards on, cache off", cache=False),
        Cell(
            "bare_window1",
            "bare with output.streaming.first_window_chars 1 (no first-window holdback)",
            guards=False,
            cache=False,
            first_window_chars=1,
        ),
        Cell("cache", "exact cache on (miss path), guards off", guards=False),
        Cell("full", "default config: guards + exact cache, in-process limits"),
        Cell("full_redis", "default config with redis (limits, budgets, cache, spend guard)", redis=True),
        Cell(
            "full_ml", "default config + ML guards (presidio pii, toxicity, topic) via GG_MODELS_DIR", ml=True
        ),
    )
}


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen[bytes]
    log: Path

    def stop(self, timeout: float = 30) -> None:
        if self.popen.poll() is not None:
            return
        self.popen.send_signal(signal.SIGINT)
        try:
            self.popen.wait(timeout)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            self.popen.wait()


@dataclass
class Session:
    out: Path
    args: argparse.Namespace
    procs: list[Proc] = field(default_factory=lambda: [])

    def start(self, name: str, cmd: Sequence[str], env: Mapping[str, str], cwd: Path) -> Proc:
        log = self.out / f"{name}.log"
        handle = log.open("ab")
        popen = subprocess.Popen(cmd, env=dict(env), cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)  # noqa: S603
        handle.close()
        proc = Proc(name, popen, log)
        self.procs.append(proc)
        return proc

    def stop_all(self) -> None:
        for proc in reversed(self.procs):
            proc.stop()
        self.procs.clear()


def base_env() -> dict[str, str]:
    keep = ("PATH", "HOME", "TMPDIR", "LANG", "TIKTOKEN_CACHE_DIR")
    return {k: os.environ[k] for k in keep if k in os.environ}


def http_get(url: str, timeout: float = 2) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def http_post(url: str, body: Mapping[str, Any], headers: Mapping[str, str]) -> tuple[int, bytes]:
    req = urllib.request.Request(  # noqa: S310
        url, data=orjson.dumps(body), headers={"content-type": "application/json", **headers}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.status, resp.read()


def wait_ready(url: str, proc: Proc, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.popen.poll() is not None:
            raise RuntimeError(f"{proc.name} exited early; see {proc.log}")
        with contextlib.suppress(OSError):
            http_get(url)
            return
        time.sleep(0.2)
    raise RuntimeError(f"{proc.name} not ready after {timeout}s; see {proc.log}")


def env_info() -> dict[str, Any]:
    cpu = platform.processor()
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        cpu = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    import locust

    return {
        "cpu": cpu,
        "cores": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30),
        "os": f"{platform.system()} {platform.release()} ({platform.platform()})",
        "python": platform.python_version(),
        "locust": locust.__version__,
    }


def cell_config(cell: Cell, dest: Path) -> Path:
    config = dest / "config"
    shutil.copytree(ROOT / "config", config)
    policy = config / "policies" / "default.yaml"
    doc = yaml.safe_load(policy.read_text())
    if not cell.guards:
        doc["defaults"]["mode"] = "off"
        doc["input"]["allow_unredacted_upstream"] = True
        for side in ("input", "output"):
            for guard in doc[side]["guards"]:
                guard["mode"] = "off"
    if cell.first_window_chars is not None:
        doc["output"]["streaming"]["first_window_chars"] = cell.first_window_chars
    policy.write_text(yaml.safe_dump(doc, sort_keys=False))
    if not cell.cache:
        path = config / "cache.yaml"
        doc = yaml.safe_load(path.read_text())
        doc["exact"]["enabled"] = False
        doc["semantic"]["enabled"] = False
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return config


def gateway_env(cell: Cell, config: Path, redis_url: str) -> dict[str, str]:
    env = {
        **base_env(),
        "GG_ENV": "dev",
        "GG_PORT": str(GATEWAY_PORT),
        "GG_CONFIG_DIR": str(config),
        "GG_KEYS_FILE": str(KEYS_FILE),
        "GG_MODEL_PROFILE": "loadtest",
        "GG_PROVIDERS__MOCK_HTTP__BASE_URL": f"{MOCK_URL}/v1",
        "GG_LANGFUSE__SAMPLE_RATE": "0",
    }
    if cell.redis:
        env["GG_REDIS_URL"] = redis_url
    if cell.ml:
        env["GG_MODELS_DIR"] = str(ROOT / ".models")
    return env


class Sampler:
    """1 Hz cpu% and rss of the gateway, the mock and the locust process tree (workers included)"""

    def __init__(self, pids: Mapping[str, int]) -> None:
        self._roots = {name: psutil.Process(pid) for name, pid in pids.items()}
        # cpu_percent is a delta since the previous call on the same object, so objects are kept per pid
        self._known: dict[int, psutil.Process] = {}
        self.samples: dict[str, list[tuple[float, float, float]]] = {name: [] for name in pids}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _tree(self, root: psutil.Process) -> list[psutil.Process]:
        try:
            found = [root, *root.children(recursive=True)]
        except psutil.Error:
            return []
        out: list[psutil.Process] = []
        for p in found:
            if p.pid not in self._known:
                self._known[p.pid] = p
                with contextlib.suppress(psutil.Error):
                    p.cpu_percent(None)
                continue
            out.append(self._known[p.pid])
        return out

    def _loop(self) -> None:
        for root in self._roots.values():
            self._tree(root)
        while not self._stop.wait(1.0):
            for name, root in self._roots.items():
                cpu = top = rss = 0.0
                for p in self._tree(root):
                    with contextlib.suppress(psutil.Error):
                        one = p.cpu_percent(None)
                        cpu += one
                        top = max(top, one)
                        rss += p.memory_info().rss / 2**20
                self.samples[name].append((cpu, top, rss))

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, rows in self.samples.items():
            # the first and last second include spawn and shutdown
            steady = rows[1:-1] or rows
            cpus = [c for c, _, _ in steady]
            out[f"{name}_cpu_mean"] = sum(cpus) / len(cpus) if cpus else None
            out[f"{name}_cpu_max"] = max(cpus) if cpus else None
            # the busiest single process: the per-process ceiling is what matters for locust
            out[f"{name}_proc_cpu_max"] = max((t for _, t, _ in steady), default=None)
            out[f"{name}_rss_max_mb"] = max((r for _, _, r in rows), default=None)
        return out


@contextlib.contextmanager
def locust_proc(session: Session, cmd: Sequence[str], env: Mapping[str, str], name: str) -> Iterator[Proc]:
    proc = session.start(name, cmd, env, ROOT)
    try:
        yield proc
    finally:
        try:
            proc.popen.wait(session.args.warmup + session.args.duration + 60)
        except subprocess.TimeoutExpired:
            proc.stop()
        session.procs.remove(proc)


def run_locust(
    session: Session,
    *,
    cell: str,
    scenario: str,
    user_class: str,
    host: str,
    api_key: str,
    target_rps: float,
    per_user_rate: float,
    duration: float,
    warmup: float,
    kind: str,
    gateway: Proc | None,
    mock: Proc,
    processes: int = 1,
) -> dict[str, Any]:
    tag = f"{cell}-{scenario}-{target_rps:g}" + ("" if kind == "fixed" else f"-{kind}")
    run_dir = session.out / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    users = max(1, round(target_rps / per_user_rate))
    started = time.time()
    record_after = started + warmup
    env = {
        **base_env(),
        "GG_LT_API_KEY": api_key,
        "GG_LT_MODEL": "gg/weak" if api_key else "gpt-mock",
        "GG_LT_RATE": str(per_user_rate),
        "GG_LT_SAMPLES": str(run_dir),
        "GG_LT_RECORD_AFTER": str(record_after),
    }
    cmd = [
        str(VENV_BIN / "locust"),
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "-u",
        str(users),
        "-r",
        str(max(users, 1)),
        "-t",
        f"{int(warmup + duration)}s",
        "--host",
        host,
        "--csv",
        str(run_dir / "locust"),
        "--only-summary",
        "--stop-timeout",
        "10",
        "--loglevel",
        "WARNING",
    ]
    if processes > 1:
        cmd += ["--processes", str(processes)]
    cmd.append(user_class)
    mock_before = orjson.loads(http_get(f"{MOCK_URL}/_stats"))["requests"]
    metrics_before = overhead_metric() if gateway is not None else {}
    with locust_proc(session, cmd, env, f"locust-{tag}") as proc:
        pids = {"locust": proc.popen.pid, "mock": mock.popen.pid}
        if gateway is not None:
            pids["gateway"] = gateway.popen.pid
        with Sampler(pids) as sampler:
            proc.popen.wait(warmup + duration + 60)
    ended = time.time()
    rows = report.load_samples(run_dir)
    for path in run_dir.glob("samples-*.csv"):
        with path.open("rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
    measured = max(ended - record_after - 0.5, 1.0)
    result: dict[str, Any] = {
        "cell": cell,
        "scenario": scenario,
        "kind": kind,
        "target_rps": target_rps,
        "users": users,
        "per_user_rate": per_user_rate,
        "measured_s": round(measured, 1),
        "client": report.summarise_client(rows, min(measured, duration)),
        "resources": sampler.summary(),
        "upstream_calls": orjson.loads(http_get(f"{MOCK_URL}/_stats"))["requests"] - mock_before,
    }
    if gateway is not None:
        records = report.read_request_logs(gateway.log, record_after, ended)
        result["server"] = report.summarise_server(records)
        result["metric_overhead_mean_ms"] = metric_delta(metrics_before, overhead_metric())
    c = result["client"]
    print(
        f"  {tag:<32} rps={report.fmt(c['rps'])} err={100 * (c['error_rate'] or 0):.2f}% "
        f"p50={report.fmt(c['total_ms']['p50'])} p99={report.fmt(c['total_ms']['p99'])} "
        f"ttft50={report.fmt(c['ttft_ms']['p50'])}",
        flush=True,
    )
    return result


def overhead_metric() -> dict[str, float]:
    """sum and count of gg_gateway_overhead_seconds per (stream, phase), from /metrics"""
    out: dict[str, float] = {}
    for line in http_get(f"{GATEWAY_URL}/metrics", timeout=10).decode().splitlines():
        if not line.startswith(("gg_gateway_overhead_seconds_sum{", "gg_gateway_overhead_seconds_count{")):
            continue
        name_labels, _, value = line.rpartition(" ")
        name, _, labels = name_labels.partition("{")
        parts = dict(kv.split("=", 1) for kv in labels.rstrip("}").replace('"', "").split(","))
        out[f"{parts['stream']}/{parts['phase']}/{name.rsplit('_', 1)[1]}"] = float(value)
    return out


def metric_delta(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    """mean overhead per (stream, phase) over the run, in ms"""
    out: dict[str, float] = {}
    for key, total in after.items():
        if not key.endswith("/sum"):
            continue
        base = key.removesuffix("/sum")
        count = after.get(f"{base}/count", 0) - before.get(f"{base}/count", 0)
        if count > 0:
            out[base] = round((total - before.get(key, 0)) / count * 1000, 3)
    return out


def warm_gateway(api_key: str) -> float:
    """first request pays lazy imports and tokenizer loads; returns its latency in ms"""
    headers = {"authorization": f"Bearer {api_key}"}
    msg = [{"role": "user", "content": "warm up"}]
    t0 = time.perf_counter()
    http_post(f"{GATEWAY_URL}/v1/chat/completions", {"model": "gg/weak", "messages": msg}, headers)
    cold = (time.perf_counter() - t0) * 1000
    for i in range(20):
        body = {
            "model": "gg/weak",
            "messages": [{"role": "user", "content": f"warm {i}"}],
            "stream": i % 2 == 1,
        }
        http_post(f"{GATEWAY_URL}/v1/chat/completions", body, headers)
    return cold


@contextlib.contextmanager
def redis_db(url: str) -> Iterator[None]:
    """refuses a non-empty db, and flushes it afterwards so runs never touch someone else's keys"""
    from redis import Redis

    client = Redis.from_url(url)
    try:
        if client.dbsize():
            raise RuntimeError(f"{url} is not empty; pick an unused db with --redis-url")
        try:
            yield
        finally:
            client.flushdb()
    finally:
        client.close()


def start_gateway(session: Session, cell: Cell, *, profile_to: Path | None = None) -> tuple[Proc, float]:
    cell_dir = session.out / "cells" / f"{cell.name}{'-profile' if profile_to else ''}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    config = cell_config(cell, cell_dir)
    env = gateway_env(cell, config, session.args.redis_url)
    if profile_to is None:
        cmd = [str(VENV_BIN / "gg"), "serve"]
    else:
        cmd = [sys.executable, "-m", "cProfile", "-o", str(profile_to), "-m", "gg", "serve"]
    proc = session.start(f"gateway-{cell.name}{'-profile' if profile_to else ''}", cmd, env, cell_dir)
    wait_ready(f"{GATEWAY_URL}/readyz", proc, timeout=180)
    return proc, warm_gateway(TOKENS["loadtest"])


def fixed_runs(session: Session, cell: str, gateway: Proc | None, mock: Proc) -> list[dict[str, Any]]:
    a = session.args
    plan = [("nonstream", "NonStreamUser", a.rate, 2.0), ("stream", "StreamUser", a.stream_rate, 1.0)]
    if cell in a.low_rate_cells or (gateway is None and a.low_rate_cells):
        plan += [("nonstream", "NonStreamUser", a.low_rate, 1.0), ("stream", "StreamUser", a.low_rate, 1.0)]
    if gateway is not None and cell in a.cache_hit_cells:
        plan.append(("cachehit", "CacheHitUser", a.rate, 2.0))
    return [
        run_locust(
            session,
            cell=cell,
            scenario=scenario,
            user_class=user_class,
            host=GATEWAY_URL if gateway else MOCK_URL,
            api_key=TOKENS["loadtest"] if gateway else "",
            target_rps=rps,
            per_user_rate=per_user,
            duration=a.duration,
            warmup=a.warmup,
            kind="fixed",
            gateway=gateway,
            mock=mock,
        )
        for scenario, user_class, rps, per_user in plan
    ]


def knee_steps(session: Session, cell: str, gateway: Proc | None, mock: Proc) -> list[dict[str, Any]]:
    a = session.args
    steps: list[dict[str, Any]] = []
    for rps in a.knee_steps:
        step = run_locust(
            session,
            cell=cell,
            scenario="nonstream",
            user_class="NonStreamUser",
            host=GATEWAY_URL if gateway else MOCK_URL,
            api_key=TOKENS["loadtest"] if gateway else "",
            target_rps=rps,
            per_user_rate=2,
            duration=a.knee_duration,
            warmup=3,
            kind="knee",
            gateway=gateway,
            mock=mock,
            processes=a.locust_processes,
        )
        steps.append(step)
        c = step["client"]
        if (c["error_rate"] or 0) >= 0.01 or (c["rps"] or 0) < 0.9 * rps:
            break
    return steps


def run_burst(session: Session) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "loadtest" / "burst_limits.py"),
        "http",
        "--base-url",
        GATEWAY_URL,
        "--api-key",
        TOKENS["loadtest-limited"],
        "--model",
        "gg/weak",
        "--rpm",
        "120",
        "--users",
        "50",
        "--duration",
        "15",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False, cwd=ROOT)  # noqa: S603
    try:
        result = orjson.loads(done.stdout)
    except orjson.JSONDecodeError:
        result = {"error": done.stderr[-2000:]}
    result["exit_code"] = done.returncode
    return result


def profile_text(path: Path, limit: int = 25) -> str:
    out: list[str] = []
    for sort in ("tottime", "cumulative"):
        buf = StringIO()
        stats = pstats.Stats(str(path), stream=buf)
        stats.strip_dirs().sort_stats(sort).print_stats(limit)
        text = buf.getvalue()
        start = text.find("   ncalls")
        out += [f"Top {limit} by {sort}:", "", "```", text[start:].rstrip(), "```", ""]
    return "\n".join(out)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="ci perf smoke: full cell, 2 short runs, no knee")
    p.add_argument(
        "--cells",
        nargs="+",
        default=["bare", "bare_window1", "guards", "cache", "full", "full_redis"],
        choices=sorted(CELLS),
    )
    p.add_argument("--rate", type=float, default=100, help="offered rps for the fixed non-stream runs")
    p.add_argument("--stream-rate", type=float, default=50, help="offered rps for the fixed stream runs")
    p.add_argument("--duration", type=float, default=30, help="measured seconds per fixed run")
    p.add_argument("--warmup", type=float, default=5, help="discarded seconds per fixed run")
    p.add_argument("--knee-cells", nargs="*", default=["bare", "full"])
    p.add_argument("--knee-steps", nargs="+", type=float, default=[100, 200, 300, 400, 500, 600, 800])
    p.add_argument("--knee-duration", type=float, default=15)
    p.add_argument("--locust-processes", type=int, default=4, help="locust worker processes for knee steps")
    p.add_argument("--cache-hit-cells", nargs="*", default=["full", "full_redis"])
    p.add_argument("--low-rate", type=float, default=10, help="offered rps of the near-idle runs")
    p.add_argument("--low-rate-cells", nargs="*", default=["bare", "full"])
    p.add_argument("--skip-knee", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--skip-burst", action="store_true")
    p.add_argument(
        "--redis-url", default="redis://localhost:6379/15", help="an empty db; flushed after the run"
    )
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.smoke:
        a.cells, a.knee_cells, a.cache_hit_cells, a.low_rate_cells = ["full"], [], [], []
        a.rate, a.stream_rate, a.duration, a.warmup = 20, 10, 8, 2
        a.skip_knee = a.skip_profile = a.skip_burst = True
    return a


def main(argv: Sequence[str] | None = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    # absolute: the gateway runs with its working directory inside the bundle
    out = (a.out or ROOT / "loadtest" / "results" / f"{stamp}{'-smoke' if a.smoke else ''}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = Session(out, a)
    runs: list[dict[str, Any]] = []
    extras: dict[str, Any] = {"knee": {}, "cold_start_ms": {}}
    failed = False
    try:
        mock = session.start(
            "mock",
            [
                str(VENV_BIN / "uvicorn"),
                "mock_upstream.app:app",
                "--port",
                str(MOCK_PORT),
                "--loop",
                "uvloop",
                "--http",
                "httptools",
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            {**base_env(), **MOCK_ENV},
            ROOT,
        )
        wait_ready(f"{MOCK_URL}/healthz", mock)
        print(f"results -> {out}", flush=True)
        print("cell direct (mock baseline)", flush=True)
        runs += fixed_runs(session, "direct", None, mock)
        for name in a.cells:
            cell = CELLS[name]
            print(f"cell {name}: {cell.description}", flush=True)
            with redis_db(a.redis_url) if cell.redis else contextlib.nullcontext():
                gateway, cold = start_gateway(session, cell)
                extras["cold_start_ms"][name] = round(cold, 1)
                try:
                    runs += fixed_runs(session, name, gateway, mock)
                    if not a.skip_knee and name in a.knee_cells:
                        extras["knee"][name] = knee_steps(session, name, gateway, mock)
                    if not a.skip_burst and cell.redis:
                        extras["burst"] = run_burst(session)
                finally:
                    gateway.stop()
                    session.procs.remove(gateway)
        if not a.skip_knee and a.knee_cells:
            print("direct knee check at the top gateway step", flush=True)
            top = max((s[-1]["target_rps"] for s in extras["knee"].values() if s), default=0)
            if top:
                a.knee_steps = [top]
                extras["knee"]["direct"] = knee_steps(session, "direct", None, mock)
        if not a.skip_profile:
            print("profile: full cell under cProfile", flush=True)
            prof = out / "gateway.prof"
            gateway, _ = start_gateway(session, CELLS["full"], profile_to=prof)
            try:
                run_locust(
                    session,
                    cell="full-profiled",
                    scenario="nonstream",
                    user_class="NonStreamUser",
                    host=GATEWAY_URL,
                    api_key=TOKENS["loadtest"],
                    target_rps=a.rate,
                    per_user_rate=2,
                    duration=a.duration,
                    warmup=a.warmup,
                    kind="profile",
                    gateway=gateway,
                    mock=mock,
                )
            finally:
                gateway.stop()
                session.procs.remove(gateway)
            extras["profile"] = profile_text(prof)
    except Exception:
        # keep whatever was measured: the partial bundle is still written below
        failed = True
        traceback.print_exc()
    finally:
        session.stop_all()

    meta = {
        "started": stamp,
        "env": env_info(),
        "mock": ", ".join(f"{k.removeprefix('MOCK_').lower()}={v}" for k, v in MOCK_ENV.items()),
        "warmup_s": a.warmup,
        "duration_s": a.duration,
        "cells": {n: CELLS[n].description for n in a.cells},
        "cold_start_ms": extras["cold_start_ms"],
        "redis_url": a.redis_url if any(CELLS[n].redis for n in a.cells) else None,
    }
    (out / "summary.json").write_bytes(
        orjson.dumps({"meta": meta, "runs": runs, "extras": extras}, option=orjson.OPT_INDENT_2)
    )
    (out / "report.md").write_text(report.render(meta, runs, extras))
    for log in out.glob("gateway-*.log"):
        with log.open("rb") as src, gzip.open(f"{log}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        log.unlink()
    print(f"report: {out / 'report.md'}", flush=True)
    bad = [
        r
        for r in runs
        if r["kind"] == "fixed"
        and r["scenario"] != "cachehit"
        and (r["client"]["requests"] == 0 or (r["client"]["error_rate"] or 0) >= 0.01)
    ]
    for r in bad:
        print(
            f"FAIL {r['cell']}-{r['scenario']}: {r['client']['requests']} requests, "
            f"error rate {r['client']['error_rate']}",
            file=sys.stderr,
        )
    return 1 if bad or failed else 0


if __name__ == "__main__":
    sys.exit(main())
