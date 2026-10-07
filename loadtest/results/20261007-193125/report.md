# GG load test - 20261007-193125

Host: Apple M4, 10 cores, 16 GB RAM, Darwin 25.6.0 (macOS-26.6.2-arm64-arm-64bit-Mach-O), Python 3.13.11, locust 2.46.7. Local processes, no containers, no CPU pinning; the load generator shares the host.

Conditions: gateway `gg serve`, 1 uvicorn worker (uvloop, httptools), model profile `loadtest`, Langfuse export off, ML guards off unless the cell says so. Mock upstream, 1 worker: ttft_ms=50, itl_ms=5, output_tokens=40. Fixed-rate runs: 5 s warmup (discarded) + 30 s measured, open-ish arrivals (constant_throughput users with random start phase). Percentiles come from raw per-request samples. First request after start (ms): {'bare': 442.3, 'bare_window1': 429.3, 'guards': 414.3, 'cache': 406.7, 'full': 389.0, 'full_redis': 410.5}.

Cells: `bare` guards off, cache off, no redis; `bare_window1` bare with output.streaming.first_window_chars 1 (no first-window holdback); `guards` rule-based input/output guards on, cache off; `cache` exact cache on (miss path), guards off; `full` default config: guards + exact cache, in-process limits; `full_redis` default config with redis (limits, budgets, cache, spend guard)

## Gateway overhead

Client delta: gateway percentile minus direct-to-mock percentile at the same offered rate (non-stream: total latency; stream: time to first content chunk). Server: the gateway's own per-request `timing_ms` from the `request.completed` log (non-stream `overhead` = total - upstream; stream `ttft_added`, and `overhead` = ttft_added + stream_tail). Server-Timing `gw` is the response header as the client saw it; the metric column is the mean of `gg_gateway_overhead_seconds` (phase total) over the run.

| cell | scenario | RPS | client delta p50/p95/p99 ms | server p50/p95/p99 ms | stream overhead p50/p99 ms | Server-Timing gw p50/p99 ms | gg_gateway_overhead_seconds mean ms |
|---|---|---|---|---|---|---|---|
| bare | nonstream | 100 | 20.4 / 35.2 / 30.5 | 9.2 / 30.0 / 32.2 | - | 9.1 / 32.2 | 12.67 |
| bare | stream | 50 | 85.1 / 103.2 / 124.6 | 50.7 / 62.7 / 67.1 | 70.1 / 83.4 | 49.8 / 65.3 | 64.12 |
| bare | nonstream | 10 | 6.3 / 14.8 / 22.9 | 3.2 / 6.6 / 7.2 | - | 2.9 / 7.0 | 3.86 |
| bare | stream | 10 | 52.9 / 57.2 / 69.5 | 44.1 / 47.9 / 50.1 | 47.2 / 53.5 | 43.6 / 49.7 | 46.86 |
| bare_window1 | nonstream | 100 | 21.3 / 35.1 / 47.4 | 9.4 / 26.7 / 41.7 | - | 9.3 / 41.8 | 11.45 |
| bare_window1 | stream | 50 | 37.9 / 57.1 / 80.9 | 11.0 / 25.5 / 29.2 | 26.9 / 50.8 | 10.3 / 28.3 | 20.95 |
| guards | nonstream | 100 | 33.1 / 101.0 / 248.4 | 25.9 / 58.6 / 187.9 | - | 25.8 / 187.9 | 31.68 |
| guards | stream | 50 | 109.6 / 182.0 / 425.7 | 84.7 / 145.9 / 399.3 | 102.4 / 439.0 | 83.5 / 390.9 | 104.19 |
| cache | nonstream | 100 | 5.5 / 28.9 / 57.4 | 6.8 / 28.2 / 50.9 | - | 6.6 / 50.9 | 11.26 |
| cache | stream | 50 | 83.4 / 200.7 / 285.5 | 50.3 / 74.1 / 170.7 | 67.0 / 199.5 | 49.5 / 168.7 | 66.00 |
| full | nonstream | 100 | 28.1 / 80.8 / 127.9 | 22.0 / 63.2 / 113.0 | - | 21.9 / 113.2 | 25.79 |
| full | stream | 50 | 106.8 / 128.9 / 169.0 | 81.2 / 98.7 / 118.2 | 98.3 / 133.8 | 80.2 / 116.9 | 88.73 |
| full | nonstream | 10 | 23.1 / 46.2 / 54.7 | 25.0 / 42.2 / 48.1 | - | 24.8 / 47.9 | 23.35 |
| full | stream | 10 | 64.2 / 88.0 / 132.6 | 55.2 / 61.0 / 129.1 | 58.7 / 133.9 | 54.4 / 128.7 | 58.10 |
| full_redis | nonstream | 100 | 14.9 / 62.6 / 131.6 | 16.8 / 52.7 / 106.0 | - | 16.8 / 109.3 | 23.24 |
| full_redis | stream | 50 | 106.4 / 172.4 / 215.4 | 89.5 / 107.4 / 163.4 | 106.1 / 177.4 | 88.8 / 162.2 | 97.40 |

## All runs

| cell | scenario | offered RPS | achieved RPS | requests | errors | latency p50/p95/p99 ms | TTFT p50/p99 ms | server overhead p50/p99 ms | gw CPU % mean | gw RSS MB | locust CPU % max (1 proc) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct | nonstream | 100 | 100.1 | 3011 | 0.00% | 279.3 / 290.4 / 301.9 | - | - | - | - | 6.3 |
| direct | stream | 50 | 51.2 | 1500 | 0.00% | 287.0 / 297.6 / 304.8 | 61.5 / 68.4 | - | - | - | 15.2 |
| direct | nonstream | 10 | 10.3 | 300 | 0.00% | 289.7 / 295.2 / 296.4 | - | - | - | - | 0.9 |
| direct | stream | 10 | 10.3 | 300 | 0.00% | 290.0 / 298.7 / 301.4 | 53.3 / 56.6 | - | - | - | 3.7 |
| bare | nonstream | 100 | 99.9 | 3008 | 0.00% | 299.7 / 325.6 / 332.4 | - | 9.19 / 32.24 | 41.0 | 128 | 2.1 |
| bare | stream | 50 | 51.3 | 1500 | 0.00% | 347.0 / 380.3 / 392.3 | 146.6 / 193.0 | 70.06 / 83.41 | 21.7 | 101 | 5.7 |
| bare | nonstream | 10 | 10.2 | 300 | 0.00% | 296.0 / 310.0 / 319.3 | - | 3.24 / 7.20 | 12.8 | 101 | 1.5 |
| bare | stream | 10 | 10.3 | 300 | 0.00% | 301.4 / 310.5 / 315.7 | 106.1 / 126.1 | 47.15 / 53.53 | 12.5 | 92 | 3.9 |
| bare_window1 | nonstream | 100 | 100.1 | 3015 | 0.00% | 300.7 / 325.6 / 349.3 | - | 9.39 / 41.65 | 41.3 | 128 | 2.1 |
| bare_window1 | stream | 50 | 51.3 | 1500 | 0.00% | 329.5 / 354.5 / 361.4 | 99.4 / 149.3 | 26.91 / 50.79 | 21.9 | 102 | 5.1 |
| guards | nonstream | 100 | 100.4 | 3000 | 0.00% | 312.4 / 391.4 / 550.3 | - | 25.95 / 187.93 | 46.7 | 132 | 3.4 |
| guards | stream | 50 | 51.2 | 1500 | 0.00% | 359.5 / 487.3 / 583.6 | 171.1 / 494.0 | 102.39 / 438.96 | 26.7 | 103 | 5.4 |
| cache | nonstream | 100 | 100.0 | 3009 | 0.00% | 284.8 / 319.4 / 359.2 | - | 6.76 / 50.86 | 42.3 | 126 | 2.8 |
| cache | stream | 50 | 51.3 | 1500 | 0.00% | 338.8 / 470.3 / 621.0 | 144.9 / 353.8 | 67.01 / 199.49 | 24.1 | 108 | 5.5 |
| full | nonstream | 100 | 99.9 | 3013 | 0.00% | 307.4 / 371.3 / 429.8 | - | 21.99 / 113.04 | 46.1 | 117 | 3.4 |
| full | stream | 50 | 51.3 | 1500 | 0.00% | 364.2 / 389.8 / 434.4 | 168.3 / 237.4 | 98.26 / 133.76 | 24.0 | 111 | 5.0 |
| full | nonstream | 10 | 10.3 | 300 | 0.00% | 312.8 / 341.4 / 351.1 | - | 24.99 / 48.09 | 12.2 | 107 | 0.8 |
| full | stream | 10 | 10.3 | 300 | 0.00% | 302.9 / 326.7 / 372.9 | 117.4 / 189.3 | 58.72 / 133.88 | 12.5 | 103 | 2.6 |
| full | cachehit | 100 | 101.1 | 3050 | 0.00% | 53.6 / 90.0 / 152.0 | - | - / - | 12.6 | 94 | 2.8 |
| full_redis | nonstream | 100 | 99.9 | 3004 | 0.00% | 294.2 / 353.1 / 433.4 | - | 16.83 / 105.96 | 46.8 | 144 | 2.8 |
| full_redis | stream | 50 | 51.3 | 1500 | 0.00% | 364.6 / 396.8 / 474.0 | 167.9 / 283.8 | 106.13 / 177.37 | 25.7 | 109 | 4.4 |
| full_redis | cachehit | 100 | 99.8 | 3005 | 0.00% | 45.6 / 56.3 / 106.1 | - | 57.90 / 57.90 | 17.0 | 108 | 3.4 |

## Stage time p50 (ms, non-stream, from the request log)

| stage | bare @100 | bare @10 | bare_window1 @100 | guards @100 | cache @100 | full @100 | full @10 | full_redis @100 |
|---|---|---|---|---|---|---|---|---|
| auth | 0.008 | 0.039 | 0.008 | 0.007 | 0.008 | 0.007 | 0.019 | 0.007 |
| authorize | 0.006 | 0.023 | 0.006 | 0.005 | 0.006 | 0.005 | 0.011 | 0.005 |
| cache_exact | 0.009 | 0.037 | 0.009 | 0.009 | 0.055 | 0.050 | 0.121 | 0.886 |
| guard_in_pre | 0.029 | 0.124 | 0.028 | 8.729 | 0.028 | 3.785 | 10.534 | 2.042 |
| guard_out | 0.021 | 0.071 | 0.020 | 5.485 | 0.020 | 6.644 | 5.018 | 4.788 |
| limits | 0.054 | 0.301 | 0.053 | 0.050 | 0.055 | 0.047 | 0.159 | 0.949 |
| observability | 0.005 | 0.024 | 0.005 | 0.004 | 0.005 | 0.004 | 0.012 | 0.004 |
| parse | 0.023 | 0.108 | 0.022 | 0.020 | 0.023 | 0.021 | 0.056 | 0.020 |
| probes | 0.006 | 0.023 | 0.006 | 0.006 | 0.009 | 0.008 | 0.018 | 0.008 |
| probes.router | 0.002 | 0.007 | 0.002 | 0.002 | 0.002 | 0.002 | 0.004 | 0.002 |
| probes.semantic_cache | - | - | - | - | 0.002 | 0.001 | 0.003 | 0.001 |
| routing | 0.008 | 0.031 | 0.008 | 0.007 | 0.008 | 0.007 | 0.016 | 0.007 |

## Throughput steps - `bare`

Knee = highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the first step's p99.

| offered RPS | achieved RPS | errors | p50/p95/p99 ms | gw CPU % mean | locust CPU % max (1 proc) |
|---|---|---|---|---|---|
| 100 | 100.0 | 0.00% | 288.1 / 316.1 / 463.6 | 41.9 | 4.9 |
| 200 | 200.2 | 0.00% | 338.3 / 361.4 / 400.0 | 62.0 | 5.5 |
| 300 | 298.7 | 0.00% | 434.4 / 518.5 / 556.9 | 87.3 | 6.1 |
| 400 | 328.0 | 0.00% | 618.3 / 719.1 / 738.7 | 84.9 | 6.3 |

Knee: **299 RPS** (offered 300)

## Throughput steps - `full`

Knee = highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the first step's p99.

| offered RPS | achieved RPS | errors | p50/p95/p99 ms | gw CPU % mean | locust CPU % max (1 proc) |
|---|---|---|---|---|---|
| 100 | 100.2 | 0.00% | 296.5 / 380.2 / 551.1 | 45.4 | 5.4 |
| 200 | 192.4 | 0.00% | 415.2 / 656.2 / 784.8 | 75.9 | 5.8 |
| 300 | 233.7 | 0.00% | 613.5 / 876.3 / 918.7 | 82.0 | 5.6 |

Knee: **192 RPS** (offered 200)

## Throughput steps - `direct`

Knee = highest step with achieved >= 95% of offered, errors < 1% and p99 <= 2x the first step's p99.

| offered RPS | achieved RPS | errors | p50/p95/p99 ms | gw CPU % mean | locust CPU % max (1 proc) |
|---|---|---|---|---|---|
| 400 | 399.8 | 0.00% | 279.8 / 301.4 / 318.1 | - | 9.2 |

Knee: **400 RPS** (offered 400)

## Burst against a limited key

```json
{
  "requests": 12468,
  "status": {
    "429": 12419,
    "200": 49
  },
  "peak_inflight": 50,
  "spent_usd": 0.00441,
  "violations": [],
  "exit_code": 0
}
```

## Mock upstream calls per run

| run | upstream calls |
|---|---|
| bare nonstream @100 | 3500 |
| bare stream @50 | 1750 |
| bare nonstream @10 | 350 |
| bare stream @10 | 350 |
| bare_window1 nonstream @100 | 3500 |
| bare_window1 stream @50 | 1750 |
| guards nonstream @100 | 3449 |
| guards stream @50 | 1750 |
| cache nonstream @100 | 3500 |
| cache stream @50 | 1750 |
| full nonstream @100 | 3500 |
| full stream @50 | 1750 |
| full nonstream @10 | 350 |
| full stream @10 | 350 |
| full cachehit @100 | 20 |
| full_redis nonstream @100 | 3500 |
| full_redis stream @50 | 1750 |
| full_redis cachehit @100 | 21 |

## Profile (cProfile, main thread, non-stream run)

Top 25 by tottime:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    9.966    9.966   42.401   42.401 runners.py:86(run)
   153177    1.548    0.000    1.548    0.000 {method 'poll' of 'select.poll' objects}
   435240    1.451    0.000    3.404    0.000 main.py:316(model_construct)
  1168850    0.561    0.000    1.045    0.000 _fields.py:705(resolve_default_value)
   150930    0.472    0.000    6.021    0.000 stream.py:115(feed)
6656846/6586563    0.445    0.000    0.649    0.000 {built-in method builtins.isinstance}
278802/275292    0.424    0.000    1.138    0.000 _readers.py:156(__call__)
1439403/1183550    0.407    0.000    2.369    0.000 {built-in method builtins.next}
   157950    0.396    0.000    0.999    0.000 sse.py:41(_consume)
  4069582    0.394    0.000    0.394    0.000 {method 'get' of 'dict' objects}
   463320    0.376    0.000    0.592    0.000 usage.py:29(get_path)
   282312    0.361    0.000    6.450    0.000 http11.py:194(_receive_event)
   312465    0.348    0.000    0.348    0.000 {method 'search' of 're.Pattern' objects}
   143910    0.335    0.000    2.528    0.000 stream.py:189(_choice)
1072346/137987    0.302    0.000   26.541    0.000 runner.py:42(_call)
   241704    0.277    0.000    0.574    0.000 _asyncio.py:1396(receive)
   715629    0.268    0.000    0.331    0.000 contextlib.py:108(__init__)
     7030    0.262    0.000    0.272    0.000 pii_patterns.py:116(find)
   153231    0.261    0.000    0.500    0.000 _sockets.py:143(extra_attributes)
   196599    0.257    0.000    0.274    0.000 timeouts.py:50(reschedule)
   282312    0.253    0.000    2.519    0.000 _connection.py:438(next_event)
     7020    0.251    0.000    3.277    0.000 connection_pool.py:254(_assign_requests_to_connections)
  1330071    0.250    0.000    0.406    0.000 <frozen importlib._bootstrap>:645(parent)
    94801    0.249    0.000    0.438    0.000 dataclasses.py:1599(_replace)
   241704    0.246    0.000    2.967    0.000 anyio.py:27(read)
```

Top 25 by cumulative:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
   1261/1    0.050    0.000   43.238   43.238 {built-in method builtins.exec}
        1    0.000    0.000   43.238   43.238 <frozen runpy>:201(run_module)
        1    0.000    0.000   43.208   43.208 <frozen runpy>:65(_run_code)
        1    0.000    0.000   43.208   43.208 __main__.py:1(<module>)
        1    0.000    0.000   42.999   42.999 main.py:17(main)
        1    0.000    0.000   42.997   42.997 serve.py:19(run_serve)
        1    0.000    0.000   42.995   42.995 main.py:503(run)
        1    0.000    0.000   42.407   42.407 server.py:85(run)
        1    0.000    0.000   42.402   42.402 runners.py:160(run)
        1    9.966    9.966   42.401   42.401 runners.py:86(run)
   137987    0.033    0.000   26.576    0.000 runner.py:36(run)
1072346/137987    0.302    0.000   26.541    0.000 runner.py:42(_call)
   137987    0.032    0.000   26.487    0.000 stage.py:42(__call__)
934359/137987    0.159    0.000   26.437    0.000 runner.py:50(call_next)
   137987    0.038    0.000   26.359    0.000 stage.py:75(__call__)
   137987    0.041    0.000   25.710    0.000 stages.py:78(__call__)
   134477    0.030    0.000   24.636    0.000 stage.py:34(__call__)
   130977    0.041    0.000   23.862    0.000 stages.py:97(__call__)
   130977    0.027    0.000   23.414    0.000 probes.py:66(__call__)
   130977    0.026    0.000   23.240    0.000 stage.py:25(__call__)
   130977    0.032    0.000   23.094    0.000 executor.py:70(__call__)
   130977    0.045    0.000   23.053    0.000 executor.py:84(_run)
   130977    0.083    0.000   22.975    0.000 executor.py:167(_attempt)
544028/281957    0.187    0.000   22.604    0.000 streams.py:34(__anext__)
   278807    0.212    0.000   21.824    0.000 executor.py:277(_next_chunk)
```


## Findings and optimisation backlog

Written after the run from this bundle, `../20261007-195300` (ML guards cell at 20 RPS) and targeted before/after
experiments on the same machine and day.

### Headline (local processes, mock upstream, Apple M4)

| Metric | Value | Conditions |
|---|---|---|
| Gateway overhead, bare, server p50 / p99 | 9.2 / 32.2 ms | non-stream, 100 RPS; 3.2 / 7.2 ms at 10 RPS |
| Gateway overhead, default config, server p50 / p99 | 22.0 / 113.0 ms | non-stream, 100 RPS, rule guards + exact cache (miss) |
| Same with redis | 16.8 / 106.0 ms | `full_redis` |
| Client-observed delta vs direct, p50 | bare 20.4 ms, default 28.1 ms | includes the extra local hop |
| TTFT added, stream, server p50 / p99 | bare 50.7 / 67.1, default 81.2 / 118.2 ms | 50 RPS, mock ITL 5 ms |
| TTFT added without first-window holdback | 11.0 / 29.2 ms | `bare_window1` |
| Exact-cache hit, client p50 / p99 | 53.6 / 152.0 ms (45.6 / 106.1 with redis) | vs ~280 ms upstream; server 36 ms p50, of which input guards 19 ms |
| Max sustainable RPS (1 worker) | bare 299, default ~192 | knee rule above; gateway CPU 87% / 76% |
| ML guards (`full_ml`), server p50 / p99 | 41.1 / 76.3 ms; stream TTFT added 114.5 ms | 20 RPS; tier-2 probe 19.5 ms, presidio 10.1 ms |
| Rate limit burst | 49 admitted of 12,468 (bound 50), every 429 well-formed, 0 violations | 120 rpm key, 50 users, 15 s, redis |
| Gateway RSS | 92-144 MB | ML guards off |
| First request after start | 389-442 ms | lazy imports, tokenizer load |

### Where the time goes

1. **Per-chunk upstream processing is ~85% of gateway CPU** (profile: `executor._next_chunk` 22.6 s of 26.5 s
   pipeline time). Every request, non-stream included, reads upstream as SSE and builds pydantic chunk models:
   `model_construct` 435k calls / 3.4 s with 1.17M `resolve_default_value`, openai_compat `stream.feed` 6.0 s,
   h11 event parsing 6.5 s, an `asyncio.timeout` per chunk. This is what caps the knee at ~300 RPS per worker.
2. **Event-loop round trips inside stages, not CPU, dominate stage latency under load.** `ConcurrentProbesStage`
   spent 2.6-4.4 ms p50 (20-33 ms p99) creating and awaiting tasks for probes that finish instantly. The guard engine
   has the same pattern per tier: `guard_in_pre` is 0.6 ms of CPU serially but 8-19 ms p50 at 100 RPS, which is
   most of the cache-hit latency.
3. **Stream first-window holdback.** The output guard holds text until 40 chars arrive, even when every output
   guard is off: ~40 ms of TTFT at 5 ms ITL (50.7 -> 11.0 ms with a 1-char window). At real ITLs (15-30 ms) this
   is 100-200 ms.
4. **Gen-2 GC pauses** of 27-53 ms every 5-8 s at 100 RPS (gc callbacks in the gateway). `gc.freeze()` after
   start-up cut them to 7-9 ms; p99 overhead did not move outside run-to-run noise at this load.
5. **Low load is slower per request than moderate load** (stage times 3-4x higher at 10 RPS): macOS moves a mostly
   idle process onto efficiency cores. Compare like with like.

### Done (measured)

- **Eager-start probe tasks** (`src/gg/pipeline/probes.py`): probes that finish without suspending no longer pay
  task-scheduling, `wait` and `gather` hops. bare, non-stream, 100 RPS, 3 runs each, same gateway build otherwise:

  | metric | before (3 runs) | after (3 runs) | median change |
  |---|---|---|---|
  | `probes` stage p50 | 3.15 / 2.59 / 4.39 ms | 0.006 / 0.006 / 0.006 ms | -99.8% |
  | `probes` stage p99 | 33.1 / 20.8 / 21.3 ms | 0.024 / 0.032 / 0.032 ms | -99.9% |
  | server overhead p50 | 12.08 / 9.98 / 13.77 ms | 11.83 / 7.53 / 9.26 ms | -23% |
  | server overhead p95 | 35.5 / 31.4 / 36.6 ms | 31.0 / 28.9 / 28.2 ms | -19% |

  Accepted: the stage itself is gone from the profile; end-to-end gain exceeds 5% but is close to the run-to-run
  spread at p50, so the p95 and stage numbers are the stronger evidence.

### Backlog, in priority order (not done: outside `observability/`, `pipeline/`, `core/`)

1. **Eager-start guard tiers** in `GuardrailEngine._run_tier` (`src/gg/guardrails/engine.py`, C6/C7), the same
   change as the probes stage. Expected: `guard_in_pre` 8-19 ms -> ~1 ms p50 under load, cache-hit p50 ~50 -> ~20 ms.
   Do not switch the whole loop to `asyncio.eager_task_factory`: tried via a loop hook, it breaks anyio cancel
   scopes in httpcore (every request 500s).
2. **Skip the first-window holdback when no streaming output detector is active**, and revisit
   `first_window_chars: 40` (`src/gg/guardrails/output/stream_guard.py`, C7). Measured -40 ms TTFT p50 at 5 ms ITL.
3. **Cheaper chunk path** (C3/C4, plan O6/O13): construct chunk models without per-field default resolution (or
   relay bytes when nothing needs rewriting), one deadline per stream instead of `asyncio.timeout` per chunk.
   Target: knee +30-50%, CPU per request -30%.
4. **ML guards** (plan O1/O3): the tier-2 probe (topic embedding) costs 19.5 ms p50 on the critical path and
   presidio 10 ms with a 2-thread CPU executor; size the executor to cores and share the embedding with the
   semantic cache.
5. **Redis round trips** (plan O5): `cache_exact` 0.9 ms and `limits` 0.95 ms p50 with local redis vs 0.05 ms
   in-process; pipeline the cache GET with the limiter script.
6. **`gc.freeze()` after start-up** in the app lifespan (plan O17, `src/gg/app/`): 5x shorter full collections,
   tail effect unproven at 100 RPS; cheap to add.

Confirmed: uvloop and httptools are active (`gg serve` sets both; plan O12).
