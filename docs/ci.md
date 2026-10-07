# CI and the eval gate

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. No job sees a provider key,
downloads model weights or calls a paid API.

| Job | What it runs |
|---|---|
| `lint` | `ruff check .` and `ruff format --check .` |
| `types` | `pyright` (strict, `src/`) |
| `imports` | `lint-imports` (layer and independence contracts in `pyproject.toml`) |
| `tests` | `pytest` without the `ml` extra, with a `redis:8` service and `GG_TEST_REDIS_URL` so the real-redis tests run |
| `eval-gate` | `python -m gg.evalgate`: guardrail and cache evals against committed baselines |

## What the eval gate covers

- **Guardrails** (`evals/guardrails/`): every item runs in process through the real guard engine and policy
  (`config/policies/default.yaml`); output items also go through `StreamGuard` in chunks. Only rule-based guards
  decide: `GG_MODELS_DIR` is unset and no remote clients are built, so model-backed guards (toxicity, topic,
  grounding, presidio) and PromptGuard allow everything. Their numbers come from local runs with the `ml` extra.
- **Cache** (`evals/cache/`): every pair goes through the production semantic tags, the `num_sig` filter and
  the in-memory index, scored with the deterministic hashing embedder at the threshold in `suite.yaml`. This
  catches changes to tagging and filtering, not embedding quality; bge-small precision comes from
  `python -m gg.cache.eval --embedder fastembed`.
- Not covered yet: conformance, routing and perf suites, the held-out slice, wiring items through the app.

Each suite's `suite.yaml` lists its gates, so loosening one is a visible diff:

- `item_regressions`: an item that was `caught`/`allowed` (guardrails) or a `no_hit` pair that was `rejected`
  (cache) in the baseline and is now `missed`/`false_positive`/`false_hit`. New items never regress.
- Metric gates: `min`, `max`, or `max_drop` against the baseline value. `mode: gate` fails the job,
  `mode: report` only shows `warn`. The cache precision floor is report-only because the hashing embedder is
  not the production one.

On a pull request the baselines and `accepted_changes.yaml` are read at the merge base with `main`; on a push
to `main` at the previous head. Editing `baseline.json` in the same change therefore cannot hide a regression,
and an old `accepted_changes.yaml` entry cannot excuse a new one.

## Running it locally

```sh
.venv/bin/python -m gg.evalgate                          # both suites against working-tree baselines
.venv/bin/python -m gg.evalgate --base-ref origin/main   # what ci does on a pull request
.venv/bin/python -m gg.evalgate --suite cache --update-baselines
```

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
4. Commit it as `fix: stop flagging mid-sentence mentions of "ignore previous instructions"`, push and open the
   pull request against `main`.
5. Expected: the `ci / eval-gate` check fails with `item_regressions: fail`; the job summary lists the item
   under "Item regressions". With `eval-gate` a required check in the `main` ruleset, GitHub shows
   "Merging is blocked".
6. Screenshot (a) the checks list with `eval-gate` failing, (b) the job summary table, and (c) a follow-up
   commit that reverts the anchoring (or narrows it correctly) turning the check green.
7. Close the pull request without deleting it, and link it from the README as a demonstration pull request
   with a real CI run. Do not edit the baseline, the summary HTML or the gate to produce the failure.
