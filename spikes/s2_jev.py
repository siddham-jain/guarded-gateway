"""spike s2: measure jev latency and answer shape from this machine.

usage: python3 spikes/s2_jev.py [warm_rounds=3]   (stdlib only; reads GG_JEV_API_KEY from .env)
"""

import base64
import csv
import http.client
import json
import os
import socket
import ssl
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

HOST = "api.typesafe.ai"
MODEL = "jev-1.13.0"
ATTEMPT_MS = 400
DEADLINE_MS = 600
PRICE_PER_MTOK = 0.042
SPIKES = Path(__file__).resolve().parent
OUT = SPIKES / "out"

PROMPTS = [
    ("p01", "small", "hi there!", []),
    ("p02", "small", "What is the capital of Australia?", []),
    ("p03", "small", "Translate to French: 'The meeting is moved to Thursday.'", []),
    ("p04", "small", "Fix the grammar: 'me and him goes to the store yesterday'", []),
    ("p05", "small", "Extract every email address from: contact [EMAIL_1] or [EMAIL_2] for billing.", []),
    (
        "p06",
        "small",
        "yes, go ahead",
        ["Should I rename the column `created` to `created_at` in the users table?"],
    ),
    ("p07", "mid", "Write a polite email declining a meeting invite because of a scheduling conflict.", []),
    ("p08", "mid", "Explain how HTTPS certificate validation works, for a junior developer.", []),
    (
        "p09",
        "mid",
        "Write a Python function that returns the n most common words in a text file, ignoring case and punctuation.",
        [],
    ),
    (
        "p10",
        "mid",
        "Summarise the pros and cons of PostgreSQL versus MySQL for a small SaaS app in a short table.",
        [],
    ),
    ("p11", "mid", "A shirt costs $40 after a 20% discount. What was the original price?", []),
    ("p12", "mid", "Write a SQL query that returns each customer's total spend in 2025, highest first.", []),
    (
        "p13",
        "frontier",
        "Our Celery workers randomly hang after ~2 hours with no error. Redis broker, prefetch 4, acks_late on. What could cause this and how do I debug it?",
        ["We moved from RabbitMQ to Redis last week."],
    ),
    (
        "p14",
        "frontier",
        "Design a rate limiter for a multi-region API gateway that must allow 1,000 req/s per key globally with at most 5% overshoot. Compare token bucket in Redis vs local buckets with periodic sync.",
        [],
    ),
    (
        "p15",
        "frontier",
        "Refactor this React component tree so form state lives in a reducer and validation runs on blur; explain the migration across the 6 files involved.",
        [],
    ),
    (
        "p16",
        "frontier",
        "Our p99 latency doubled after upgrading from Python 3.12 to 3.13 but p50 is unchanged. Walk through how to find the cause.",
        [],
    ),
    (
        "p17",
        "frontier",
        "Write a 1,500-word explainer on how central bank balance-sheet reduction affects long-term bond yields, with historical examples.",
        [],
    ),
    ("p18", "frontier_reasoning", "Prove that there are infinitely many primes of the form 4k+3.", []),
    (
        "p19",
        "frontier_reasoning",
        "Find all integer solutions to x^2 - 7y^2 = 1 with 0 < x < 10,000 and explain why the method finds all of them.",
        [],
    ),
    (
        "p20",
        "frontier_reasoning",
        "Given n intervals, find the minimum number of points such that every interval contains at least one point, then prove your greedy algorithm is optimal.",
        [],
    ),
]


def load_env() -> None:
    env_file = SPIKES.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())


def ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


class Connection:
    """one tls connection to jev, with the setup phases timed."""

    def __init__(self) -> None:
        self.dns_ms: float | None = None
        start = time.perf_counter()
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            self.via = "proxy"
            raw = self._connect_via_proxy(urlsplit(proxy))
        else:
            self.via = "direct"
            address = socket.getaddrinfo(HOST, 443, type=socket.SOCK_STREAM)[0][4]
            resolved = time.perf_counter()
            self.dns_ms = ms(start, resolved)
            start = resolved
            raw = socket.create_connection(address[:2], timeout=10)
        connected = time.perf_counter()
        context = ssl.create_default_context()
        context.set_alpn_protocols(["http/1.1"])
        self.sock = context.wrap_socket(raw, server_hostname=HOST)
        handshaken = time.perf_counter()
        self.connect_ms = ms(start, connected)
        self.tls_ms = ms(connected, handshaken)
        self.open = True

    @staticmethod
    def _connect_via_proxy(proxy) -> socket.socket:
        raw = socket.create_connection((proxy.hostname, proxy.port), timeout=10)
        lines = [f"CONNECT {HOST}:443 HTTP/1.1", f"Host: {HOST}:443"]
        if proxy.username:
            credentials = f"{unquote(proxy.username)}:{unquote(proxy.password or '')}"
            lines.append("Proxy-Authorization: Basic " + base64.b64encode(credentials.encode()).decode())
        raw.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            chunk = raw.recv(4096)
            if not chunk:
                raise ConnectionError("proxy closed the connection during CONNECT")
            reply += chunk
        status_line = reply.split(b"\r\n", 1)[0]
        if b" 200" not in status_line:
            raise ConnectionError(f"proxy refused CONNECT: {status_line.decode(errors='replace')}")
        return raw

    def request(self, method: str, path: str, body: bytes = b"") -> dict:
        head = [
            f"{method} {path} HTTP/1.1",
            f"Host: {HOST}",
            "User-Agent: gg-spike-s2/0",
            "Accept: application/json",
            f"Authorization: Bearer {os.environ['GG_JEV_API_KEY']}",
        ]
        if body:
            head += ["Content-Type: application/json", f"Content-Length: {len(body)}"]
        start = time.perf_counter()
        sent = first_byte = None
        try:
            self.sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode() + body)
            sent = time.perf_counter()
            response = http.client.HTTPResponse(self.sock, method=method)
            response.begin()
            first_byte = time.perf_counter()
            data = response.read()
            done = time.perf_counter()
        except (OSError, http.client.HTTPException) as error:
            # a stall or reset is a latency sample too; record it and drop the connection
            failed = time.perf_counter()
            self.close()
            phase = "send" if sent is None else "wait" if first_byte is None else "body"
            return {
                "status": 0,
                "error": f"{type(error).__name__} during {phase}",
                "headers": {},
                "data": b"",
                "send_ms": ms(start, sent or failed),
                "wait_ms": ms(sent, first_byte or failed) if sent else 0.0,
                "download_ms": ms(first_byte, failed) if first_byte else 0.0,
                "request_ms": ms(start, failed),
            }
        if response.will_close:
            self.close()
        return {
            "status": response.status,
            "headers": {name.lower(): value for name, value in response.getheaders()},
            "data": data,
            "send_ms": ms(start, sent),
            "wait_ms": ms(sent, first_byte),
            "download_ms": ms(first_byte, done),
            "request_ms": ms(start, done),
        }

    def close(self) -> None:
        self.open = False
        self.sock.close()


def score_body(request: str, recent_turns: list[str], questions: dict) -> bytes:
    state: dict = {"request": request}
    if recent_turns:
        state["recent_user_turns"] = recent_turns
    state["context"] = {"conversation_depth": "short" if recent_turns else "new", "tools_offered": "none"}
    return json.dumps({"model": MODEL, "state": state, "questions": questions}).encode()


def sample(
    kind: str, prompt_id: str, round_no: int, result: dict, connection: Connection | None = None
) -> dict:
    row = {
        "kind": kind,
        "prompt_id": prompt_id,
        "round": round_no,
        "status": result["status"],
        "dns_ms": connection.dns_ms if connection else None,
        "connect_ms": connection.connect_ms if connection else None,
        "tls_ms": connection.tls_ms if connection else None,
        "send_ms": result["send_ms"],
        "wait_ms": result["wait_ms"],
        "download_ms": result["download_ms"],
        "total_ms": result["request_ms"],
        "request_id": result["headers"].get("x-typesafe-request-id"),
        "edge": result["headers"].get("cf-ray", "-").rsplit("-", 1)[-1],
        "server_timing": result["headers"].get("server-timing"),
        "model": None,
        "input_tokens": None,
        "tier": None,
        "error": result.get("error"),
    }
    if connection:
        row["total_ms"] = round(
            sum(row[k] or 0 for k in ("dns_ms", "connect_ms", "tls_ms")) + result["request_ms"], 1
        )
    if kind.endswith("score") and result["status"] == 200:
        payload = json.loads(result["data"])
        row["model"] = payload.get("model")
        row["input_tokens"] = payload.get("usage", {}).get("input_tokens")
        row["tier"] = payload.get("answers", {}).get("tier", {}).get("choice")
    elif result["status"] != 200:
        reason = result.get("error") or result["data"][:200]
        print(
            f"  {kind} {prompt_id} round {round_no}: status {result['status']} after {result['request_ms']} ms: {reason!r}",
            file=sys.stderr,
        )
    return row


def save_fixture(prompt_id: str, expected: str, body: bytes, result: dict) -> None:
    fixtures = OUT / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    kept_headers = {k: v for k, v in result["headers"].items() if k not in {"set-cookie", "date"}}
    fixture = {
        "expected_tier": expected,
        "request": json.loads(body),
        "status": result["status"],
        "headers": kept_headers,
        "response": json.loads(result["data"])
        if result["status"] == 200
        else result["data"].decode(errors="replace"),
    }
    (fixtures / f"{prompt_id}.json").write_text(json.dumps(fixture, indent=2) + "\n")


def run(warm_rounds: int) -> list[dict]:
    questions = json.loads((SPIKES / "qset-v1.json").read_text())
    rows: list[dict] = []

    print(f"cold: {len(PROMPTS)} calls, each on a fresh connection")
    for prompt_id, _, request, turns in PROMPTS:
        connection = Connection()
        result = connection.request("POST", "/v1/systemone", score_body(request, turns, questions))
        rows.append(sample("cold_score", prompt_id, 0, result, connection))
        if connection.open:
            connection.close()

    print(f"warm: {warm_rounds} rounds x {len(PROMPTS)} prompts on one kept-alive connection")
    connection = Connection()
    connection.request("GET", "/v1/models")
    reconnects = 0
    for round_no in range(1, warm_rounds + 1):
        for prompt_id, expected, request, turns in PROMPTS:
            body = score_body(request, turns, questions)
            for kind, method, path, payload in (
                ("warm_edge", "GET", "/cdn-cgi/trace", b""),
                ("warm_origin", "GET", "/v1/models", b""),
                ("warm_score", "POST", "/v1/systemone", body),
            ):
                if not connection.open:
                    connection = Connection()
                    reconnects += 1
                result = connection.request(method, path, payload)
                rows.append(sample(kind, prompt_id, round_no, result))
                if kind == "warm_score" and round_no == 1 and result["status"] == 200:
                    save_fixture(prompt_id, expected, body, result)
    if connection.open:
        connection.close()
    print(f"warm reconnects: {reconnects}")
    return rows


def percentile(values: list[float], q: int) -> float:
    if len(values) < 2:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": round(statistics.fmean(values), 1),
        "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0.0,
    }


def share_over(rows: list[dict], kind: str, limit_ms: int) -> float:
    """percent of calls slower than limit_ms; a failed call counts as slower."""
    calls = [r for r in rows if r["kind"] == kind]
    slow = sum(r["status"] != 200 or r["total_ms"] > limit_ms for r in calls)
    return round(100 * slow / len(calls), 1)


def summarise(rows: list[dict], trace: str) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    totals = {
        kind: [r["total_ms"] for r in ok if r["kind"] == kind]
        for kind in ("cold_score", "warm_score", "warm_origin", "warm_edge")
    }
    cold = [r for r in ok if r["kind"] == "cold_score"]
    warm_score = totals["warm_score"]
    tokens = [r["input_tokens"] for r in ok if r["input_tokens"]]
    location = dict(line.split("=", 1) for line in trace.splitlines() if "=" in line)
    return {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "client_country": location.get("loc"),
        "cloudflare_edge": location.get("colo"),
        "connection_path": "proxy" if os.environ.get("HTTPS_PROXY") else "direct",
        "calls": {"total": len(rows), "failed": len(rows) - len(ok)},
        "latency_ms": {kind: describe(values) for kind, values in totals.items() if values},
        "cold_phases_p50_ms": {
            phase: percentile([r[phase] for r in cold if r[phase] is not None], 50)
            for phase in ("dns_ms", "connect_ms", "tls_ms", "send_ms", "wait_ms", "download_ms")
            if any(r[phase] is not None for r in cold)
        },
        "breakdown_p50_ms": {
            "edge_round_trip": percentile(totals["warm_edge"], 50),
            "edge_to_origin_extra": round(
                percentile(totals["warm_origin"], 50) - percentile(totals["warm_edge"], 50), 1
            ),
            "jev_scoring_extra": round(percentile(warm_score, 50) - percentile(totals["warm_origin"], 50), 1),
            "new_connection_extra": round(
                percentile(totals["cold_score"], 50) - percentile(warm_score, 50), 1
            ),
        },
        "budget": {
            f"{prefix}_over_{limit_name}_{limit}ms_pct": share_over(rows, f"{prefix}_score", limit)
            for prefix in ("warm", "cold")
            for limit_name, limit in (("attempt", ATTEMPT_MS), ("deadline", DEADLINE_MS))
        },
        "failures": [
            {key: r[key] for key in ("kind", "prompt_id", "round", "status", "error", "total_ms")}
            for r in rows
            if r["status"] != 200
        ],
        "usage": {
            "input_tokens_mean": round(statistics.fmean(tokens)) if tokens else None,
            "input_tokens_total": sum(tokens),
            "cost_usd": round(sum(tokens) * PRICE_PER_MTOK / 1e6, 6),
        },
        "response_models": sorted({r["model"] for r in ok if r["model"]}),
        "edges_seen": sorted({r["edge"] for r in ok}),
        "request_id_header": "x-typesafe-request-id" if any(r["request_id"] for r in ok) else None,
        "server_timing_seen": any(r["server_timing"] for r in ok),
    }


def main() -> None:
    load_env()
    if not os.environ.get("GG_JEV_API_KEY"):
        sys.exit("GG_JEV_API_KEY is not set (put it in .env at the repo root)")
    probe = Connection()
    check = probe.request("GET", "/v1/models")
    trace = probe.request("GET", "/cdn-cgi/trace")["data"].decode()
    probe.close()
    if check["status"] != 200:
        sys.exit(f"key check failed: HTTP {check['status']} {check['data'][:200]!r}")

    rows = run(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    run_dir = OUT / time.strftime("run-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    with open(run_dir / "samples.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows, trace)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
