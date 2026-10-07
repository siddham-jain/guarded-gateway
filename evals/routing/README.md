# Routing eval set and harness

The routing eval measures how well GG's router (Jev "system one", `gg/auto`) separates prompts the weak model
handles from prompts that need the strong model. It produces a cost–quality curve, APGR against random and
heuristic baselines, and the α presets for `config/routing.yaml` (plan: C5 §10, C11 §3.8).

Everything paid for is done once and cached; every sweep, report and α change afterwards is free.

## Files

| Path | What |
|---|---|
| `items/prompts.jsonl` | the hand-written eval set (versioned; its sha256 is in every report) |
| `judge/pairwise_v1.md` | the pairwise judge prompt; its file name and content hash form the judge version |
| `suite.yaml` | frozen model pairs (`ci`, `dev-free`, `openrouter`): weak, strong and judge model + params |
| `runs/<pair>/` | stores and reports of one pair (`runs/dry-run/` for the dry run) |

Code: `src/gg/routing/eval/` (items, graders, judge, stores, budget, runner, sweep, report) and the CLI wiring
in `src/gg/cli/routing_eval.py`, which builds GG's gateway app in process.

## The eval set

170 hand-written prompts, short on purpose (median 113 characters, longest 265) to keep generation cheap. No PII, no
secrets, nothing copied from public benchmarks. Each line:

```json
{"id": "rt042", "category": "math_hard", "split": "tune", "source": "handwritten", "expected_tier": "strong",
 "messages": [{"role": "user", "content": "..."}], "grader": {"type": "numeric", "answer": 130},
 "reference": "Answer: 130", "max_tokens": 1000}
```

| Category | n | author prior weak / strong | Graders |
|---|---|---|---|
| chit_chat | 10 | 10 / 0 | judge 4, constraints 5, regex 1 |
| simple_qa | 20 | 20 / 0 | exact 12, numeric 8 |
| extraction | 18 | 12 / 6 | json 18 |
| classification | 10 | 8 / 2 | exact 10 (two sarcasm / root-cause items) |
| rewrite | 12 | 10 / 2 | constraints 6, regex 5, exact 1 |
| coding_routine | 12 | 12 / 0 | exact 8 (output prediction), regex 4 |
| coding_hard | 16 | 0 / 16 | exact 14 (Python gotchas), regex 2 |
| math_easy | 12 | 12 / 0 | numeric 12 |
| math_hard | 20 | 0 / 20 | numeric 20 |
| reasoning | 16 | 3 / 13 | numeric 9, exact 6, regex 1 |
| knowledge_hard | 6 | 0 / 6 | choice 6 |
| writing | 10 | 4 / 6 | judge 10 |
| multi_turn | 8 | 5 / 3 | regex 3, numeric 2, constraints 1, judge 2 |
| **total** | **170** | **96 / 74** | 154 checker-graded, 16 judge-graded |

- `expected_tier` is the author's guess, written before any model or Jev output was seen. It is never the
  routing label; it only feeds the dry-run fakes and a side metric ("agreement with the authors' prior").
- `split`: 103 `tune` / 67 `heldout`, stratified by category (positions 1 and 3 of every 5 per category are
  held out). α presets are chosen on `tune` and reported on `heldout`.
- Every checker item carries a `reference` that passes its own checker (enforced by a unit test). Maths and
  code-output answers were computed by running the code, not by hand.
- Prompts with an objective answer end with "End with a final line 'Answer: <answer>'" so the answer can be
  extracted without a judge.

## Scoring

### Per response: checkers (154 items, deterministic, $0)

Each checker returns 1 (pass) or 0 (fail). `<think>` blocks are stripped first; the "final answer" is the value
after the last `Answer:` line, or the whole response if there is none.

| Type | Passes when |
|---|---|
| `numeric` | the last number in the final answer is within `tol` (absolute) / `rel_tol` of `answer`; `$ € £ ,` ignored |
| `exact` (`match: exact`) | the final answer, the unfenced body, or any line (minus an `Output:` prefix) equals an accepted answer after normalisation (case, whitespace, backticks, trailing period) |
| `exact` (`match: word`) | an accepted answer appears as a whole word in the final answer **and no `reject` label does** (a hedge like "Sydney or Canberra" fails) |
| `choice` | the first standalone letter A–H in the final answer is the key |
| `regex` | every `all` pattern matches and no `none` pattern does (case-insensitive unless `case_sensitive`) |
| `json` | the first JSON value in the response (fences allowed) matches `expect` as a subset: objects by key, lists by position and length, numbers within `tol` (numeric strings accepted), strings after normalisation; `{"$any": [...]}` accepts alternatives, `{"$contains": "..."}` a substring |
| `constraints` | every listed constraint holds: word counts (`max_words`, `min_words`, `exact_words`), `max_sentences`, `lines`, `max_lines`, `list_items` (bulleted or numbered lines), `word_initial`, `contains_all/any/none`, `regex` |

For a checker item, the weak grade `g_w` and strong grade `g_s` are the two checker results.

### Per pair: pairwise judge (16 items)

Open-ended items (`grader.type: judge`) are graded by an LLM judge comparing the two answers
(`judge/pairwise_v1.md`, MT-Bench style). The judge sees the conversation, the item rubric (if any), the
reference answer (if any) and both answers, and replies `{"verdict": "A" | "B" | "tie", "reason": "..."}`. It is
told to rank correctness first, then instruction following, then helpfulness, and to ignore position, length
and confident tone.

- **Both orders**: once with the weak answer as A (`ws`), once with the strong answer as A (`sw`).
- **Final verdict**: the agreed winner if both orders agree, otherwise `tie`. An unparseable verdict counts as
  disagreement. The report shows the **position-consistency rate** (share of items where the orders agree) and
  the number of invalid verdicts.
- **Grades**: strong wins → `g_s = 1, g_w = 0`; weak wins → `1, 0`; tie → `0.5, 0.5`.
- **Judge model** comes from `suite.yaml` and must not be the strong model's family (self-preference):
  `dev-free` uses `gpt-oss-120b` on Groq to judge Gemini; `openrouter` uses Claude Haiku 4.5 to judge OpenAI.
- Judge-human agreement is not measured yet (C11 T3.5 `label` command); spot-check ~20 judged pairs by hand
  before quoting the judge-graded numbers.

### Routing label

`strong_wins = g_s > g_w`. Ties, both-correct and both-wrong are not strong wins.

### Aggregation (all on cached results; `src/gg/routing/eval/sweep.py`)

A router assigns a score `s_i`; at threshold α it sends item `i` strong iff `s_i >= α`.

- `quality(α) = mean(g_s if s_i >= α else g_w)`; `%strong(α) = mean(s_i >= α)`;
  `cost(α) = mean(cost_s if strong else cost_w) + scorer cost per request`, in USD at catalog list price
  (free tiers included at list price so the curve is comparable), reported per 1k requests.
- `PGR(α) = (quality(α) − q_weak) / (q_strong − q_weak)`; undefined (shown as —) when the strong model is not
  better on the slice.
- **Curve**: items sorted by score; one point per distinct score (tie block). Inside a tie block the curve is
  linear, which is exactly the expectation of random tie-breaking (Jev returns 2-decimal probabilities, so
  ties are common).
- **APGR** (RouteLLM code form): `(AUC_router − q_weak) / (q_strong − q_weak)` with the trapezoidal area under
  quality vs %strong over [0, 1]. Also the paper form: mean PGR at 10%, 20%, …, 100% strong.
- **CPT(50%) / CPT(80%)**: the smallest %strong whose PGR reaches 50% / 80% on the interpolated curve.
- **AUROC** of the router score against `strong_wins` (Mann-Whitney, average ranks for ties).
- **Under-routing at α**: strong-win items routed weak / all strong-win items, with the Wilson 95% upper bound.
- **95% CI** on APGR: 1,000 bootstrap resamples of items (fixed seed 7), percentile interval.
- **Presets**: on `tune`, α for `economy` / `balanced` / `quality` is the highest α (fewest strong calls) whose
  operating point reaches PGR 0.50 / 0.80 / 0.95; each is then reported on `heldout`. `balanced` is the
  candidate default for `policy.threshold`.

Baselines, all on the same cached answers:

| Router | Score |
|---|---|
| always-weak / always-strong | the curve endpoints |
| random | analytic straight line between the endpoints (APGR 0.5, CPT(p) = p); the report also shows random at the same %strong as each α |
| length/keyword heuristic | hand-set logistic over log length, code markers, maths markers, reasoning keywords and turn count (`scorers.heuristic_score`); not fitted |
| oracle | `g_s − g_w`: strong exactly where it helps most; upper bound at every budget |

## Running it

All commands from the repo root. Exit codes: 0 done, 2 config/harness error, 3 refused or stopped by the cap.

```bash
# dry run: whole pipeline through the in-process gateway on the ci profile (mock provider), fake scorer, $0
.venv/bin/python -m gg.cli.routing_eval --dry-run

# estimate a live run without calling anything
.venv/bin/python -m gg.cli.routing_eval --pair openrouter --estimate-only

# pilot: 26 items round-robin over categories, then the full set (reruns only pay for what is missing)
.venv/bin/python -m gg.cli.routing_eval --pair openrouter --limit 26 --max-usd 0.50
.venv/bin/python -m gg.cli.routing_eval --pair openrouter --max-usd 2.75

# free-tier pair (Gemini + Groq); billed spend is Jev only
.venv/bin/python -m gg.cli.routing_eval --pair dev-free --max-usd 0.10 --concurrency 2

# re-report from the stores, e.g. after changing policy.threshold; no calls
.venv/bin/python -m gg.cli.routing_eval --pair openrouter --offline
```

Options: `--phases score,generate,judge` (e.g. `--phases generate,judge` when no Jev key is set), `--limit N`,
`--alpha` (report at a different α), `--out DIR`, `--concurrency N`, `--bootstrap N`.

Needs `GG_JEV_API_KEY` for the score phase and the provider keys of the pair (`GG_PROVIDERS__OPENROUTER__API_KEY`,
or the Gemini/Groq keys) in the environment or `.env`.

### What a run does

1. **score**: every prompt goes through the production `JevScorer` (`state-v1` + `qset-v1`, pinned model) with a
   longer eval deadline. Fallback scores are not cached, so a rerun retries them.
2. **generate**: every prompt on both weak and strong, through GG's own `/v1/chat/completions` (in process,
   ephemeral virtual key, `gg: {fallback: false, cache: "off"}`), so GG's adapters, cost accounting and
   per-provider spend caps (`GG_SPEND_CAP_*`) apply. An answer served by any other deployment than the frozen
   one is rejected as contaminated and not stored.
3. **judge**: judge items in both orders.
4. **report**: grades, sweep and baselines; writes `result.json`, `report.md`, `curves.csv` and
   `cost_quality.svg` (plain SVG; matplotlib is not a dependency) into `runs/<pair>/`.

### Caching and resume

`runs/<pair>/{scores,generations,judgements}.jsonl` are append-only, content-addressed stores:

| Store | Key |
|---|---|
| scores | `sha256(scorer_version, prompt)` |
| generations | `sha256(prompt, model, params incl. max_completion_tokens)` |
| judgements | `sha256(prompt, answer A, answer B, judge model, judge params, judge prompt version, order)` |

A killed run loses at most the calls in flight; a torn last line is ignored and redone. Changing a model,
param, the judge prompt or the scorer version changes the keys, so stale results are never mixed in.

### Cost safety

- Before any call the harness prints an estimate for the uncached calls (input ≈ chars / 4 + 4 per message,
  output = 50% of the cap expected, 100% worst case) at catalog prices, and **refuses to start** if the
  expected billed spend is above `--max-usd`.
- During the run each call reserves its worst-case billed cost first; the run stops cleanly (exit 3, partial
  report written) before it could exceed `--max-usd`. Free-tier deployments (`billing: free_tier`) are priced
  in the report but never count against the cap.
- GG's own provider spend caps still apply on top.

Estimates for the full set (170 items, 32 judge calls, 170 Jev calls):

| Pair | Expected billed | Worst case billed | List price (expected) |
|---|---|---|---|
| `dev-free` (Gemini 3.1 flash-lite / 3.8 flash, Groq judge) | $0.014 (Jev only) | $0.014 | $0.54 |
| `openrouter` (gpt-6-luna / gpt-6.1-sol, Haiku 4.5 judge) | $1.32 | $2.58 | $1.32 |

Free tiers have daily request limits (Gemini 3.8 flash is the tightest); run `dev-free` with
`--concurrency 2` and rerun the next day if calls fail with 429: only the missing calls are retried.

### Config the `openrouter` pair needs

`config/models.yaml` has the `openrouter` provider but no deployments. Add (verify ids and prices against
`https://openrouter.ai/api/v1/models` first):

```yaml
  - id: openrouter/openai/gpt-6-luna
    capabilities: *openai_reasoning
    defaults: {reasoning_effort: none, max_completion_tokens: 4096}
    pricing: [{effective_from: 2026-07-01, input: 0.10, output: 0.50}]
  - id: openrouter/openai/gpt-6.1-sol
    capabilities: {<<: *openai_reasoning, tools: false, sampling_params: none,
                   effort_levels: [low, medium, high, xhigh, max], effort_clamp: {none: low, minimal: low}}
    defaults: {reasoning_effort: low, max_completion_tokens: 8192}
    reliability: {timeouts: {ttft_s: 60, inter_chunk_s: 60}, deadline_s: {non_stream: 180, stream: 300}}
    pricing: [{effective_from: 2026-09-01, input: 2.00, output: 10.00}]
  - id: openrouter/anthropic/claude-haiku-4.5
    capabilities: *open_chat
    defaults: {max_completion_tokens: 1024}
    pricing: [{effective_from: 2025-10-01, input: 1.00, output: 5.00}]
```

Any other pair: add it to `suite.yaml` with deployments that exist in the catalog for its `profile`.

## Licence and data rules

- **Jev outputs (`scores.jsonl`, the `raw` field) are for local analysis only.** TypeSafe's MCA §2.3(b)
  forbids using Jev output to train or distil another model. Never train a router or classifier on the score
  store, never publish it, and keep it out of public commits (add `evals/routing/runs/*/scores.jsonl` to
  `.gitignore`). A future local classifier may only learn from the outcome labels (`g_w`, `g_s`) and the prompt
  text.
- OpenRouter terms allow own-team use only; do not republish generations as a dataset.
- Prompts contain no personal data, so free-tier providers that may train on inputs are acceptable here.

## Limitations

- 170 items gives wide intervals (see the CIs); treat differences under ~0.1 APGR as noise.
- Hand-written prompts reflect the authors' idea of traffic; the per-category table shows where the gap comes
  from. Weak models in 2026 ace most easy items, so most of the gap sits in `math_hard`, `coding_hard` and
  `reasoning`.
- The heuristic baseline is hand-set, not fitted on `tune`.
- Judge-graded items rely on one judge without human validation yet.
- Prompts are sent to Jev without the C6 redactor in front (the set contains no PII); production scoring
  always redacts first.
