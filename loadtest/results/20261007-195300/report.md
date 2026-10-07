# GG load test - 20261007-195300

Host: Apple M4, 10 cores, 16 GB RAM, Darwin 25.6.0 (macOS-26.6.2-arm64-arm-64bit-Mach-O), Python 3.13.11, locust 2.46.7. Local processes, no containers, no CPU pinning; the load generator shares the host.

Conditions: gateway `gg serve`, 1 uvicorn worker (uvloop, httptools), model profile `loadtest`, Langfuse export off, ML guards off unless the cell says so. Mock upstream, 1 worker: ttft_ms=50, itl_ms=5, output_tokens=40. Fixed-rate runs: 5 s warmup (discarded) + 20.0 s measured, open-ish arrivals (constant_throughput users with random start phase). Percentiles come from raw per-request samples. First request after start (ms): {'bare': 435.3, 'full_ml': 452.7}.

Cells: `bare` guards off, cache off, no redis; `full_ml` default config + ML guards (presidio pii, toxicity, topic) via GG_MODELS_DIR

## Gateway overhead

Client delta: gateway percentile minus direct-to-mock percentile at the same offered rate (non-stream: total latency; stream: time to first content chunk). Server: the gateway's own per-request `timing_ms` from the `request.completed` log (non-stream `overhead` = total - upstream; stream `ttft_added`, and `overhead` = ttft_added + stream_tail). Server-Timing `gw` is the response header as the client saw it; the metric column is the mean of `gg_gateway_overhead_seconds` (phase total) over the run.

| cell | scenario | RPS | client delta p50/p95/p99 ms | server p50/p95/p99 ms | stream overhead p50/p99 ms | Server-Timing gw p50/p99 ms | gg_gateway_overhead_seconds mean ms |
|---|---|---|---|---|---|---|---|
| bare | nonstream | 20 | 20.4 / 24.0 / 37.5 | 10.5 / 15.0 / 27.2 | - | 10.3 / 27.0 | 9.49 |
| bare | stream | 10 | 56.1 / 60.2 / 61.4 | 46.5 / 50.5 / 52.3 | 51.5 / 58.3 | 45.9 / 51.8 | 50.23 |
| full_ml | nonstream | 20 | 38.6 / 66.1 / 68.4 | 41.1 / 72.1 / 76.3 | - | 40.9 / 76.2 | 44.31 |
| full_ml | stream | 10 | 117.6 / 153.7 / 158.2 | 114.5 / 153.8 / 158.1 | 115.4 / 158.8 | 114.6 / 158.0 | 108.73 |

## All runs

| cell | scenario | offered RPS | achieved RPS | requests | errors | latency p50/p95/p99 ms | TTFT p50/p99 ms | server overhead p50/p99 ms | gw CPU % mean | gw RSS MB | locust CPU % max (1 proc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | nonstream | 20 | 20.0 | 401 | 0.00% | 288.8 / 298.2 / 300.4 | - | - | - | - | 2.2 |
| direct | stream | 10 | 10.4 | 200 | 0.00% | 292.1 / 300.5 / 302.3 | 53.5 / 57.2 | - | - | - | 6.4 |
| bare | nonstream | 20 | 20.4 | 400 | 0.00% | 309.2 / 322.2 / 337.9 | - | 10.48 / 27.22 | 23.3 | 144 | 1.8 |
| bare | stream | 10 | 10.4 | 200 | 0.00% | 304.8 / 317.7 / 319.4 | 109.5 / 118.5 | 51.46 / 58.25 | 12.1 | 125 | 3.9 |
| full_ml | nonstream | 20 | 20.0 | 403 | 0.00% | 327.4 / 364.3 / 368.8 | - | 41.07 / 76.33 | 70.6 | 408 | 0.7 |
| full_ml | stream | 10 | 10.3 | 200 | 0.00% | 367.6 / 409.7 / 420.4 | 171.1 / 215.4 | 115.36 / 158.81 | 46.4 | 256 | 1.5 |

## Stage time p50 (ms, non-stream, from the request log)

| stage | bare @20 | full_ml @20 |
|---|---|---|
| auth | 0.017 | 0.017 |
| authorize | 0.010 | 0.009 |
| cache_exact | 0.023 | 0.075 |
| guard_in_pre | 0.077 | 10.116 |
| guard_out | 0.052 | 4.079 |
| limits | 0.147 | 0.100 |
| observability | 0.015 | 0.008 |
| parse | 0.050 | 0.036 |
| probes | 0.017 | 19.569 |
| probes.guardrails | - | 19.529 |
| probes.router | 0.006 | 0.002 |
| probes.semantic_cache | - | 0.002 |
| routing | 0.022 | 0.015 |

## Mock upstream calls per run

| run | upstream calls |
|---|---|
| bare nonstream @20 | 500 |
| bare stream @10 | 250 |
| full_ml nonstream @20 | 500 |
| full_ml stream @10 | 250 |
