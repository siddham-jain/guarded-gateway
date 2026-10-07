# Load tests

Measures what the gateway adds on top of an upstream, using the mock upstream (`mock_upstream/`) so no real
provider is called and nothing costs money.

| File | What it is |
|---|---|
| `run.py` | Runner. Starts the mock and the gateway, runs headless Locust per cell, writes a result bundle. |
| `locustfile.py` | Locust users: `NonStreamUser`, `StreamUser`, `CacheHitUser`. Writes raw per-request samples. |
| `report.py` | Percentiles from raw samples and gateway request logs; renders `report.md`. |
| `burst_limits.py` | Rate-limit / budget burst checker (also called by `run.py` on the redis cell). |
| `results/` | One bundle per run. |

Keys come from `tests/fixtures/keys/loadtest.yaml` (`loadtest`: limits that never trip; `loadtest-limited`:
120 rpm for the burst test). Both carry the `loadtest` tag, which turns the remote PromptGuard detectors off.

## Run

```bash
.venv/bin/python -m loadtest.run                    # standard matrix, ~25 min
.venv/bin/python -m loadtest.run --smoke            # perf smoke, ~40 s
.venv/bin/python -m loadtest.run --cells bare full --skip-knee --skip-profile --skip-burst
.venv/bin/python -m loadtest.run --cells full_ml --rate 20 --stream-rate 10 --skip-knee --skip-profile
```

Ports: mock on 9100, gateway on 8100. Every process the runner starts is stopped when it exits.
The redis cell uses `redis://localhost:6379/15` (`--redis-url`); the runner refuses a non-empty db and flushes it
afterwards. The gateway runs with its working directory inside the bundle, so the repo `.env` (provider keys,
Langfuse, PromptGuard) is never loaded; only the `GG_*` variables the runner sets reach it. Langfuse export is
off (`GG_LANGFUSE__SAMPLE_RATE=0`).

Useful flags: `--rate` / `--stream-rate` (offered RPS of the fixed runs, default 100 / 50), `--duration` /
`--warmup` (default 30 s / 5 s), `--knee-steps`, `--locust-processes`, `--low-rate` (near-idle runs, default 10),
`--out DIR`.

## Scenarios

| User | Measures |
|---|---|
| `NonStreamUser` | non-stream chat, unique prompt per request, `temperature: 0` so the exact cache does a lookup + store (miss path) when it is on |
| `StreamUser` | streaming chat with `include_usage`; records time to the first content chunk and the full stream |
| `CacheHitUser` | `temperature: 0` prompts from a pool of 20, so after the first round every request is an exact-cache hit |
| burst (`burst_limits.py http`) | 50 users against the 120 rpm key for 15 s: admitted per window <= limit, every 429 has `retry-after` and `x-ratelimit-*`, no 5xx |

Each fixed run is open-ish: `constant_throughput` users with a random start phase, so the offered rate is fixed
and arrivals are spread out. Throughput steps (`knee`) use 2 RPS per user (period above the ~300 ms request time) and 4 Locust processes.

The mock replies after `ttft_ms` + `itl_ms` per token for both streams and non-stream replies (default 50 ms,
5 ms, 40 tokens). The gateway always reads upstream as a stream, so a non-stream request through the gateway and
direct to the mock take the same upstream time.

## Cells (feature attribution)

| Cell | Config |
|---|---|
| `direct` | Locust straight at the mock: the baseline |
| `bare` | guards off, cache off, no redis |
| `bare_window1` | `bare` with `output.streaming.first_window_chars: 1` |
| `guards` | rule-based input/output guards on (no ML), cache off |
| `cache` | exact cache on (in-memory), guards off |
| `full` | default config: guards + cache, in-process limits |
| `full_redis` | `full` with `GG_REDIS_URL` (limits, budgets, cache, spend guard in redis) |
| `full_ml` | `full` with `GG_MODELS_DIR=.models` (presidio pii, toxicity, topic) — not in the default set |

Each cell gets its own copy of `config/` under `cells/<name>/config`, patched by `cell_config()` in `run.py`.

## How overhead is measured

Three independent views, all in the "Gateway overhead" table of `report.md`:

1. **Client delta.** Gateway percentile minus the direct-to-mock percentile at the same offered rate
   (non-stream: total latency; stream: time to first content chunk). Includes the extra local hop.
2. **Server side.** Per-request `timing_ms` from the gateway's `request.completed` log line: non-stream
   `overhead = total - upstream`; streams `ttft_added` and `overhead = ttft_added + stream_tail`.
3. **Header and metric.** `Server-Timing: gw` as the client saw it, and the mean of
   `gg_gateway_overhead_seconds{phase="total"}` scraped from `/metrics` before and after the run.

Percentiles are computed from raw samples (`runs/*/samples-*.csv.gz`), never from Locust's rounded histogram.

## Reading a bundle

```
results/<utc stamp>/
  report.md          tables: overhead, all runs, stage p50s, throughput steps, burst, profile
  summary.json       everything report.md is rendered from
  runs/<cell>-<scenario>-<rps>/   samples-*.csv.gz (raw) + locust_*.csv
  cells/<cell>/config/            the exact config each cell ran with
  gateway-*.log.gz   gateway stdout (request.completed lines)
  gateway.prof       cProfile of the profiled run (pstats format)
```

- A fixed run is valid when errors < 1% and achieved RPS is within a few % of offered; `locust CPU % max` well
  below 100 means the load generator was not the bottleneck.
- Knee = highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the first step's p99. The
  `direct` step at the top rate shows the mock was not the limit.
- `Mock upstream calls per run` is the number of upstream calls during the run (warmup included); for
  `cachehit` it should be about the pool size.

## Perf smoke for CI

```bash
.venv/bin/python -m loadtest.run --smoke --out out/perf-smoke
```

About 40 s, mock only, no redis, no network. Runs the direct baseline and the `full` cell: non-stream at 20 RPS and
stream at 10 RPS, 2 s warmup + 8 s each. Exit code 1 when a run has no samples or >= 1% errors.
`out/perf-smoke/summary.json` has the numbers; for a latency gate read
`runs[].server.overhead_ms.p50` (non-stream) and `runs[].server.ttft_added_ms.p50` (stream) for the `full` cell.

## Profiling

`run.py` profiles the `full` cell under `cProfile` (main thread only) and prints the top functions into the
report; open `gateway.prof` with `python -m pstats`. With py-spy installed, attach to the running gateway instead:
`py-spy record --pid <gateway pid> --rate 250 --idle -o profile.svg -d 30`.

Re-render a bundle's tables from `summary.json` with `.venv/bin/python -m loadtest.report <bundle>` (this
overwrites `report.md`, including any notes appended by hand).

## Latest results

`results/20261007-193125/report.md` (standard matrix, findings and backlog at the end) and
`results/20261007-195300/report.md` (ML guards at 20 RPS). Apple M4, 10 cores, 16 GB, macOS 26.6, local processes:

| Metric | Value |
|---|---|
| Overhead p50 / p99, bare, non-stream 100 RPS (server) | 9.2 / 32.2 ms |
| Overhead p50 / p99, default config, non-stream 100 RPS (server) | 22.0 / 113.0 ms (16.8 / 106.0 with redis) |
| TTFT added p50 / p99, default config, stream 50 RPS | 81.2 / 118.2 ms (40 ms of it is the output guard's first window) |
| Exact-cache hit p50 / p99 (client) | 53.6 / 152.0 ms |
| Max sustainable RPS, 1 worker | bare 299, default ~192 |

## Caveats

Numbers are from one machine, local processes, no CPU pinning, with Locust, the mock and the gateway sharing the
host; run-to-run spread on a laptop is a few ms at p50 and more at p99. They are labelled "local, mock upstream"
and are not the deployed instance.
