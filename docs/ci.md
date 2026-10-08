# CI and the eval gate

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. No job sees a provider key,
downloads model weights or calls a paid API.

| Job | What it runs |
|---|---|
| `lint` | `ruff check .` and `ruff format --check .` |
| `types` | `pyright` (strict, `src/`) |
| `imports` | `lint-imports` (layer and independence contracts in `pyproject.toml`) |
| `tests` | `pytest` without the `ml` extra, with a `redis:8` service and `GG_TEST_REDIS_URL` so the real-redis tests run |
| `eval-gate` | `python -m gg.evalgate`: guardrail and cache evals replayed against committed baselines |

## What the eval gate covers

The gate has two modes. `--live` runs the real thing and records what the remote and model-backed parts
answered; CI replays that recording, so it needs no key, no model weights and no network.

- **Guardrails** (`evals/guardrails/`, 126 items): every item runs in process through the real guard engine and
  policy (`config/policies/default.yaml`), including the tier-2 probe and the post-hoc guards; output items also
  go through `StreamGuard` in chunks, or the buffered path when a JSON guard applies.
  - Replay: rule packs plus the Jev injection detector answering from `cassettes/jev.json`. Model-backed guards
    (toxicity, topic, grounding, presidio) allow everything, so their items are recorded as `missed` in the
    baseline and `output_catch_rate` uses the lower `replay_min` floor.
  - Live: every guard, with the local models from `GG_MODELS_DIR` and the Jev API; rewrites the cassette.
- **Cache** (`evals/cache/`, 180 pairs): a pair is a hit when its distance is inside
  `semantic.distance_threshold` and the verifier scores it at or above `verifier.min_score`.
  - Replay: distances from `cassettes/distances.json` and Jev's verdicts from `cassettes/jev.json`, judged
    against the thresholds in today's `config/cache.yaml`. Loosening either threshold therefore shows up as
    false hits. A pair added since the recording counts as a miss until the next live run.
  - Live: embeds every pair with the production model, asks Jev about each candidate, rewrites both files.
- A cassette stores answers under a hash of the request, never the prompt. If an item, a pair or a Jev question
  changes, its entry no longer matches and the detector reports an error for that item; the summary lists these
  under "Guards that could not decide". Rerun with `--live` and commit the cassette.
- Not covered yet: conformance, routing and perf suites, wiring items through the app.

Each suite's `suite.yaml` lists its gates, so loosening one is a visible diff:

- `item_regressions`: an item that was `caught`/`allowed` (guardrails) or a `no_hit` pair that was `rejected`
  (cache) in the baseline and is now `missed`/`false_positive`/`false_hit`. New items never regress.
- Metric gates: `min`, `max`, or `max_drop` against the baseline value; `replay_min` replaces `min` on replay
  runs. `mode: gate` fails the job, `mode: report` only shows `warn`.

On a pull request the baselines and `accepted_changes.yaml` are read at the merge base with `main`; on a push
to `main` at the previous head. Editing `baseline.json` in the same change therefore cannot hide a regression,
and an old `accepted_changes.yaml` entry cannot excuse a new one.

## Running it locally

```sh
.venv/bin/python -m gg.evalgate                          # both suites against working-tree baselines
.venv/bin/python -m gg.evalgate --base-ref origin/main   # what ci does on a pull request
.venv/bin/python -m gg.evalgate --suite cache --update-baselines
.venv/bin/python -m gg.evalgate --live                   # every guard, production embedder, jev; re-records
```

`--live` reads `GG_MODELS_DIR`, `GG_JEV_API_KEY` and `GG_PROMPTGUARD_API_KEY` from the environment, makes about
150 Jev calls (well under one cent) and never writes baselines. After a live run, record the replay baseline
with `--update-baselines` and commit it with the cassettes.

Exit codes: `0` pass, `1` a gate failed, `2` harness, schema or git error. Results go to `out/eval/`
(`guardrails.json`, `cache.json`, `gate.json`, `summary.md`); in CI the summary is appended to the job summary
and `out/eval/` is uploaded as the `eval-results` artifact.

To accept a regression on purpose, append an entry to the suite's `accepted_changes.yaml` in the same change.
`from` and `to` are outcomes as shown in the summary table:

```yaml
- item: in-ben-002
  from: allowed
  to: false_positive
  reason: the new rule is right to block this; see the linked issue
  pr: 57
```

## Staging the "PR blocked by eval regression" screenshot

The screenshot must show the real gate failing on a real change (C11 §3.13). If a pull request has already
been blocked by the gate for real, use that one and skip the rest.

Otherwise open a demonstration pull request with a plausible but wrong fix for a false positive:

1. Branch from an up-to-date `main`, for example `fix/override-rule-code-comments`.
2. In `config/policies/rules/injection.v1.yaml`, rule `INJ-OVR-001`, anchor the pattern to the start of a line
   so mid-sentence mentions stop matching. Replace the start of the pattern
   `'(?i)(?<!not )(?<!n''t )(?<!never )\b(?:ignore|disregard` with
   `'(?im)^\s*(?:please\s+)?(?:ignore|disregard` and leave the rest of the line unchanged.
   The rule's own `match`/`no_match` examples still pass, so the rule-pack unit tests stay green; run
   `.venv/bin/pytest tests/unit/guardrails` to confirm only the eval gate objects.
3. Run `.venv/bin/python -m gg.evalgate --suite guardrails` and confirm it exits `1` before pushing. With the
   current items, `in-ind-001` (an override hidden in an HTML comment in a document) goes from `caught` to
   `missed` while `input_catch_rate` still passes its floor, which is the point of the item-level gate.
   In replay there is no recorded Jev answer for it, because the rule blocked it before Jev was asked; a live
   run would show the Jev detector still catching it, which is worth saying when presenting the screenshot.
4. Commit it as `fix: stop flagging mid-sentence mentions of "ignore previous instructions"`, push and open the
   pull request against `main`.
5. Expected: the `ci / eval-gate` check fails with `item_regressions: fail`; the job summary lists the item
   under "Item regressions". With `eval-gate` a required check in the `main` ruleset, GitHub shows
   "Merging is blocked".
6. Screenshot (a) the checks list with `eval-gate` failing, (b) the job summary table, and (c) a follow-up
   commit that reverts the anchoring (or narrows it correctly) turning the check green.
7. Close the pull request without deleting it, and link it from the README as a demonstration pull request
   with a real CI run. Do not edit the baseline, the summary HTML or the gate to produce the failure.
