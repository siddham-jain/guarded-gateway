# GG load test - 20261007-200154

Host: Apple M4, 10 cores, 16 GB RAM, Darwin 25.6.0 (macOS-26.6.2-arm64-arm-64bit-Mach-O), Python 3.13.11, locust 2.46.7. Local processes, no containers, no CPU pinning; the load generator shares the host.

Conditions: gateway `gg serve`, 1 uvicorn worker (uvloop, httptools), model profile `loadtest`, Langfuse export off, ML guards off unless the cell says so. Mock upstream, 1 worker: ttft_ms=50, itl_ms=5, output_tokens=40. Fixed-rate runs: 5 s warmup (discarded) + 30 s measured, open-ish arrivals (constant_throughput users with random start phase). Percentiles come from raw per-request samples. First request after start (ms): {'full': 438.8}.

Cells: `full` default config: guards + exact cache, in-process limits

## Gateway overhead

Client delta: gateway percentile minus direct-to-mock percentile at the same offered rate (non-stream: total latency; stream: time to first content chunk). Server: the gateway's own per-request `timing_ms` from the `request.completed` log (non-stream `overhead` = total - upstream; stream `ttft_added`, and `overhead` = ttft_added + stream_tail). Server-Timing `gw` is the response header as the client saw it; the metric column is the mean of `gg_gateway_overhead_seconds` (phase total) over the run.

| cell | scenario | RPS | client delta p50/p95/p99 ms | server p50/p95/p99 ms | stream overhead p50/p99 ms | Server-Timing gw p50/p99 ms | gg_gateway_overhead_seconds mean ms |
|---|---|---|---|---|---|---|---|
| full | nonstream | 100 | 43.1 / 67.6 / 86.8 | 21.9 / 49.3 / 74.6 | - | 21.8 / 74.6 | 24.61 |
| full | stream | 50 | 96.5 / 117.1 / 113.1 | 57.7 / 79.1 / 84.0 | 74.4 / 97.6 | 56.6 / 82.3 | 69.16 |
| full | nonstream | 10 | 33.2 / 38.3 / 38.2 | 17.1 / 24.9 / 27.8 | - | 16.9 / 27.7 | 15.63 |
| full | stream | 10 | 50.2 / 63.3 / 62.4 | 46.5 / 53.8 / 57.3 | 47.7 / 59.4 | 45.9 / 56.2 | 48.33 |

## All runs

| cell | scenario | offered RPS | achieved RPS | requests | errors | latency p50/p95/p99 ms | TTFT p50/p99 ms | server overhead p50/p99 ms | gw CPU % mean | gw RSS MB | locust CPU % max (1 proc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | nonstream | 100 | 99.9 | 3009 | 0.00% | 281.9 / 292.9 / 307.3 | - | - | - | - | 7.4 |
| direct | stream | 50 | 51.2 | 1500 | 0.00% | 292.1 / 302.5 / 320.9 | 63.2 / 89.6 | - | - | - | 21.5 |
| direct | nonstream | 10 | 10.3 | 300 | 0.00% | 292.5 / 299.4 / 302.1 | - | - | - | - | 1.1 |
| direct | stream | 10 | 10.3 | 300 | 0.00% | 295.4 / 302.9 / 306.7 | 55.0 / 60.8 | - | - | - | 4.5 |
| full | nonstream | 100 | 99.8 | 3010 | 0.00% | 325.0 / 360.5 / 394.1 | - | 21.88 / 74.61 | 43.4 | 138 | 1.9 |
| full | stream | 50 | 51.1 | 1500 | 0.00% | 355.6 / 379.7 / 406.9 | 159.7 / 202.7 | 74.41 / 97.57 | 26.3 | 110 | 6.0 |
| full | nonstream | 10 | 10.3 | 300 | 0.00% | 325.6 / 337.7 / 340.3 | - | 17.08 / 27.82 | 14.0 | 107 | 0.9 |
| full | stream | 10 | 10.2 | 300 | 0.00% | 303.7 / 317.4 / 321.0 | 105.2 / 123.2 | 47.73 / 59.37 | 16.1 | 71 | 3.7 |
| full | cachehit | 100 | 99.8 | 3005 | 0.00% | 49.0 / 54.9 / 106.1 | - | - / - | 14.4 | 98 | 3.0 |

## Stage time p50 (ms, non-stream, from the request log)

| stage | full @100 | full @10 |
|---|---|---|
| auth | 0.007 | 0.028 |
| authorize | 0.005 | 0.016 |
| cache_exact | 0.047 | 0.164 |
| guard_in_pre | 0.371 | 0.929 |
| guard_out | 0.173 | 0.421 |
| limits | 0.046 | 0.214 |
| observability | 0.004 | 0.020 |
| parse | 0.021 | 0.085 |
| probes | 0.007 | 0.024 |
| probes.router | 0.002 | 0.005 |
| probes.semantic_cache | 0.001 | 0.005 |
| routing | 0.007 | 0.024 |

## Mock upstream calls per run

| run | upstream calls |
|---|---|
| full nonstream @100 | 3500 |
| full stream @50 | 1750 |
| full nonstream @10 | 350 |
| full stream @10 | 350 |
| full cachehit @100 | 20 |
