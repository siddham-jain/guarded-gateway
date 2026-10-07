"""locust scenarios for the gateway and for the mock upstream directly (the baseline).

pick one user class per run; loadtest/run.py drives these headless:
  NonStreamUser   S1 non-stream chat, unique prompt per request (cache miss path when the cache is on)
  StreamUser      S2 streaming chat with include_usage; records time to first content chunk
  CacheHitUser    temperature 0 prompts from a small pool, so nearly every request is an exact-cache hit

env knobs:
  GG_LT_API_KEY       bearer token (empty when hitting the mock directly)
  GG_LT_MODEL         model sent in the body (default gg/weak)
  GG_LT_RATE          requests per second per user; 0 = closed loop with no think time
  GG_LT_POOL          prompt pool size for CacheHitUser (default 20)
  GG_LT_SAMPLES       directory for raw per-request csv samples (one file per locust process)
  GG_LT_RECORD_AFTER  epoch seconds; earlier requests are warmup and are not recorded
"""

import csv
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

import gevent
from locust import HttpUser, constant, constant_throughput, events, task
from locust.contrib.fasthttp import FastHttpUser

API_KEY = os.environ.get("GG_LT_API_KEY", "")
MODEL = os.environ.get("GG_LT_MODEL", "gg/weak")
RATE = float(os.environ.get("GG_LT_RATE", "1"))
POOL = int(os.environ.get("GG_LT_POOL", "20"))
SAMPLES_DIR = os.environ.get("GG_LT_SAMPLES", "")
RECORD_AFTER = float(os.environ.get("GG_LT_RECORD_AFTER", "0"))

PROMPT = (
    "You are helping a support team triage tickets. Summarise the following customer message in one short "
    "sentence and suggest a category from billing, shipping, account or other. Message: my parcel was "
    "marked delivered yesterday but nothing arrived and the tracking page shows no photo. Ref "
)
HEADERS = {"content-type": "application/json", **({"authorization": f"Bearer {API_KEY}"} if API_KEY else {})}
FIELDS = ("scenario", "started", "status", "ok", "total_ms", "ttft_ms", "chunks", "gw_ms", "cache")


def wait_time_for(rate: float) -> Any:
    return constant(0) if rate <= 0 else constant_throughput(rate)


def body(prompt: str, *, stream: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_completion_tokens": 256,
    }
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def server_gw_ms(header: str | None) -> str:
    """the gw entry of Server-Timing: gateway time minus the terminal (upstream) stage"""
    for part in (header or "").split(","):
        name, _, value = part.strip().partition(";dur=")
        if name == "gw":
            return value
    return ""


class Recorder:
    """raw samples, not locust's rounded histogram, so percentiles are exact"""

    def __init__(self) -> None:
        self._file: TextIO | None = None
        self._writer: Any = None

    def write(self, **row: Any) -> None:
        if not SAMPLES_DIR or row["started"] < RECORD_AFTER:
            return
        if self._writer is None:
            path = Path(SAMPLES_DIR) / f"samples-{os.getpid()}.csv"
            self._file = path.open("a", newline="")
            self._writer = csv.DictWriter(self._file, FIELDS)
            if path.stat().st_size == 0:
                self._writer.writeheader()
        self._writer.writerow(row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


RECORDER = Recorder()


@events.quitting.add_listener
def _close_recorder(**_: Any) -> None:
    RECORDER.close()


@events.test_stop.add_listener
def _flush_recorder(**_: Any) -> None:
    RECORDER.close()


def _nonstream(user: FastHttpUser, scenario: str, prompt: str) -> None:
    started = time.time()
    t0 = time.perf_counter()
    with user.client.post(
        "/v1/chat/completions",
        json=body(prompt, stream=False),
        headers=HEADERS,
        name=scenario,
        catch_response=True,
    ) as resp:
        total_ms = (time.perf_counter() - t0) * 1000
        ok = resp.status_code == 200 and b'"choices"' in (resp.content or b"")
        if not ok:
            resp.failure(f"status {resp.status_code}")
        headers = resp.headers or {}
        RECORDER.write(
            scenario=scenario,
            started=started,
            status=resp.status_code,
            ok=int(ok),
            total_ms=f"{total_ms:.3f}",
            ttft_ms="",
            chunks="",
            gw_ms=server_gw_ms(headers.get("server-timing")),
            cache=headers.get("x-gg-cache", ""),
        )


class Jittered:
    """random start phase, so constant_throughput users do not fire in lockstep bursts"""

    def on_start(self) -> None:
        if RATE > 0:
            gevent.sleep(random.uniform(0, 1 / RATE))  # noqa: S311


class NonStreamUser(Jittered, FastHttpUser):
    wait_time = wait_time_for(RATE)

    @task
    def chat(self) -> None:
        _nonstream(self, "nonstream", PROMPT + uuid.uuid4().hex)


class CacheHitUser(Jittered, FastHttpUser):
    wait_time = wait_time_for(RATE)

    @task
    def chat(self) -> None:
        _nonstream(self, "cachehit", f"{PROMPT}pool-{random.randrange(POOL)}")  # noqa: S311


class StreamUser(Jittered, HttpUser):
    # requests yields sse lines as they arrive; geventhttpclient's read blocks for a full buffer, hiding ttft
    wait_time = wait_time_for(RATE)

    @task
    def chat(self) -> None:
        started = time.time()
        t0 = time.perf_counter()
        first: float | None = None
        chunks = 0
        done = False
        with self.client.post(
            "/v1/chat/completions",
            json=body(PROMPT + uuid.uuid4().hex, stream=True),
            headers=HEADERS,
            name="stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                for line in resp.iter_lines(chunk_size=None):
                    if not line.startswith(b"data: "):
                        continue
                    if line == b"data: [DONE]":
                        done = True
                    elif b'"content":"' in line and b'"content":""' not in line:
                        first = first or time.perf_counter()
                        chunks += 1
            total_ms = (time.perf_counter() - t0) * 1000
            ok = resp.status_code == 200 and done and first is not None
            if not ok:
                resp.failure(f"status {resp.status_code} done={done} chunks={chunks}")
            RECORDER.write(
                scenario="stream",
                started=started,
                status=resp.status_code,
                ok=int(ok),
                total_ms=f"{total_ms:.3f}",
                ttft_ms="" if first is None else f"{(first - t0) * 1000:.3f}",
                chunks=chunks,
                gw_ms=server_gw_ms(resp.headers.get("server-timing")),
                cache=resp.headers.get("x-gg-cache", ""),
            )
