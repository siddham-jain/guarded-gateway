# Changelog

## [Unreleased]

### Added
- Add Jev-verified semantic cache: the vector search proposes a candidate and Jev confirms it (held-out precision 29/29, hit rate 29/29).
- Add `jev_injection` tier-2 guard (Jev noul question over the scrubbed recent turns) as the default injection detector.
- Extend the guardrail eval set to 126 items and the cache pair set to 180, both with held-out splits.
- Add `python -m gg.evalgate --live` (every guard, production embedder, Jev) and cassettes so ci replays the live run offline.
- Add routing eval harness, 170-item hand-written routing set and live OpenRouter results (APGR 0.80 held-out).
- Add Jev `score_signal` option (`strong_helps` default) and ship α = 0.25 from the routing eval.
- Add OpenRouter deployments for gpt-6-luna, gpt-6.1-sol and Claude Haiku 4.5.
- Add ci workflow (ruff, pyright, import contracts, pytest with redis 8) and replay eval gate for guardrail and cache suites.
- Add locust load tests, mock upstream latency knobs and measured overhead/throughput report.
- Add Dockerfile, compose gateway and caddy hosted profile, oracle deploy scripts and deploy workflow.
- Add per-ip failed-auth limiter, GG_MAINTENANCE kill switch and `demo` profile with an always-failing mock for the fallback demo.
- Add Anthropic-compatible `/v1/messages` ingress and `/playground`.
- Add README, architecture doc and PromptGuard circuit breaker.
- Add Langfuse tracing: one trace per request with stage spans, provider-attempt generations (usage, cost, ttft), guard findings and opt-in scrubbed content, sent as otlp/json by a bounded background exporter.
- Add self-hosted Langfuse v4 stack as the compose `langfuse` profile and `x-gg-trace-id` response header.
- Add promptguard.co remote guard (input tier 2, output post-hoc) sending only scrubbed text; default injection detector.
- Add onnx ml guards (presidio pii, granite hap toxicity, embedding topic, hhem grounding) and the tier-2 guard probe, enabled by `GG_MODELS_DIR`.
- Add prometheus metrics for cache, limits, breakers and guardrails, routing savings estimate, grafana overview dashboard, providers drill-down and alert rules.
- Add ml extra (onnxruntime, fastembed, presidio, spacy model) for local guard models and embeddings.
- Add exact and semantic response cache with placeholder-safe storage, sse replay, single-flight and fail-open redis.
- Add per-key rate limits (atomic redis lua token bucket) and micro-usd budget holds with 429/402 and x-ratelimit headers.
- Wire limits, cache and semantic probe into the pipeline; cache eval pairs and threshold sweep runner.
- Add guardrail engine, policy yaml with shadow mode, rule-based injection/secret/pii guards, reversible pii redaction and streaming output guard.
- Add guardrail eval runner skeleton with Wilson intervals and item-level baseline diff.
- Add Jev routing scorer, threshold policy, session reuse and concurrent probes stage behind `gg/auto`.
- Add routing metrics (`gg_routing_decisions_total`, `gg_router_score*`) and apply routing effort overrides per attempt.
- Add Z.ai, Alibaba and MiniMax coding-plan upstreams with shared quirk profiles and `gg/coding-plans` alias.
- Add native Anthropic Messages adapter with thinking-block store.
- Add native Gemini adapter with thought-signature store.
- Add composition root (`gg serve`, `gg config check`) wiring api, routing, executor, providers, metrics and spend guard.
- Add end-to-end walking-skeleton tests driving the real app with the stock openai sdk.
- Add static routing stage resolving models, group aliases and `gg/auto` into route plans.
- Add spend-guard adapter decorator that falls back past capped providers and books per-call cost.
- Add generic OpenAI-compatible adapter with YAML quirk profiles for 23 providers, model catalog and mock upstream.
- Add reliability executor with retries, circuit breakers, cross-provider fallback and commit point.
- Add Prometheus metrics, stage timing observer, event-loop lag sampler, request log and compose stack.
- Add cost calculation in micro-usd and per-provider spend guard (in-memory and redis).
- Add OpenAI-compatible api surface, virtual keys, sse writer and `gg keys` cli.
- Add foundation: canonical schema, normalisation, errors, request context, config loader, pipeline runner, streams.
- Add model-lab and open-model host research notes under `plans/research/`.
- Add spike S2 Jev latency script, measured results and metric glossary.
- Add master plan and 13 component plans under `plans/`.
- Add provider, router, guardrail and gateway-infra research notes under `plans/research/`.
- Add `plans/REVIEW.md` consistency review of the component plans.

### Changed
- Turn hosted PromptGuard off, grounding on and the topic guard to flag at 0.70 in the default policy (1.4.0).
- Guardrail eval runs the tier-2 probe, post-hoc guards and the buffered json path like the pipeline does.
- Start guard and probe tasks eagerly — default-config overhead p99 113 → 75 ms, stream added ttft p50 81 → 58 ms.
- Reconcile the 13 component plans with the master plan and with each other.

### Fixed
- Fix OpenRouter 404s: luna drops sampling params because `require_parameters` finds no endpoint taking them.
- Fix guard backend outages logging a traceback per request; Presidio no longer warns on unmapped spaCy labels.
- Fix middleware errors on `/v1/messages` using the OpenAI error shape.
- Fix compose pulls: MinIO moved to `cgr.dev/chainguard/minio` (Docker Hub image gone) and Grafana pinned to `12.4` (no `12` tag).
- Fix budgets failing open under load: Redis uses a blocking connection pool instead of raising at 100 busy connections.
- Fix mock providers being capped at $0 by the spend guard; they are now `free_tier`.
- Fix flaky promptguard end-to-end test that counted sampled output calls.

### Removed
- Remove the unused `tracing` extra (OpenTelemetry SDK, Langfuse SDK).
- Remove local prompt-injection models (Prompt Guard 2, ProtectAI DeBERTa), their weights and export path; injection detection is PromptGuard's.

### Decisions
- Semantic cache threshold 0.15 plus Jev verification — bge-small alone tops out near 0.7 precision at any threshold (entity swaps, negations and role swaps sit closer than real paraphrases); without the verifier nothing is served.
- PromptGuard off by default — from India it takes 2.7 s p50 against a 1 s budget and blocked 11 of 42 benign eval prompts; Jev caught 29 of 30 attacks with no false blocks at ~320 ms.
- Jev guard adds one Jev call to every request that passes tier 1 (not only `gg/auto`) — it overlaps the router and semantic probes, so `gg/auto` pays nothing extra and fixed-model requests wait ~320 ms before upstream.
- Grounding on (flag only, post-hoc) costs ~100 ms cpu per sentence when the system or tool text is 150+ chars — turn it off per key tag if the shared cpu pool saturates under load.
- Eval gate replays cassettes keyed by request hash — a changed item, pair or Jev question misses its entry and shows as a guard error until `--live` re-records.
- Jev score uses `strong_helps`, chosen on the tune split — P(frontier tiers) is 0 for 107/170 prompts; held-out APGR 0.80 vs 0.65.
- Signal is part of the Jev scorer version — an α is only valid for the signal it was tuned on.
- PromptGuard has its own 0.9 s timeout and 3-failure breaker — the engine's timeout cancels the call, which can't be told apart from a probe cancellation.
- Guard weights live on a volume and download at pinned revisions on first start — the image stays ~0.9 GB with no gated weights.
- Eval gate reads baselines and accepted_changes at the merge base — a same-PR baseline edit can't hide a regression.
- Redis client uses a 64-connection blocking pool with a 0.5 s wait — redis-py's default pool raises when full, and limits/budgets treat errors as fail-open.
- Traces are built after the last byte from gg's own timings and posted as otlp/json — no sdk, no per-chunk spans, and a dead Langfuse only drops traces (`gg_telemetry_dropped_total`).
- Langfuse is self-hosted for dev — no 50k-unit cap, and prompt text never leaves the machine; switching to Langfuse Cloud is three env vars.
- Prompt injection defaults to the hosted PromptGuard API, failing to the local rule pack — no weights to download, and its free plan (60 rpm) would make a fail-closed guard an outage.
- Remote detectors only ever see placeholder-scrubbed text, and PromptGuard's irreversible `redact` is reported as a flag — gg's reversible vault stays in charge of pii.
- ML guards stay off unless `GG_MODELS_DIR` is set — tests and CI never download weights, and the core app imports without the ml extra.
- Coding-plan subscriptions are upstreams billed as `subscription` — flat fee, so the spend guard skips them and quota 429s fall back to paid labs.
- Every non-native provider is a YAML quirk profile on one OpenAI-compatible adapter — adding a provider is config, not code.
- Jev picks the model tier; a separate host-selection step picks which provider serves an open model — mixing them would break the routing eval.
- Providers without a configured spend cap get $0 — a newly added provider can't spend until someone sets its cap.
- Keep Jev's 400/600 ms budget until S5 measures from the host — ~75% of Jev latency is distance to its servers, so the host region decides it.
- Native Anthropic and Gemini adapters over their OpenAI-compat endpoints — compat layers drop caching, safety, thinking and token details.
- Raw `httpx2` for all upstreams with GG as the only retry owner — `httpx` is unmaintained and SDK retries would nest.
- Router split into scorer + policy, Jev only for now — local classifier and OpenAI Decisions plug in without touching policy or eval.
- Never train the future local router on Jev outputs — TypeSafe's agreement bans distillation.
- Cache stores outputs with PII placeholders and restores per request — prevents cross-user PII leaks via cache.
- Host on an Oracle Always Free ARM VM with docker-compose — other free tiers can't run Redis + local models or spin down.
