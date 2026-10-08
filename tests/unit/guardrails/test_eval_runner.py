import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from gg.guardrails.eval.__main__ import main
from gg.guardrails.eval.items import load_items
from gg.guardrails.eval.stats import percentile, wilson
from tests.unit.guardrails.support import POLICY_DIR, ROOT

ITEMS = ROOT / "evals" / "guardrails" / "items"
BASELINE = ROOT / "evals" / "guardrails" / "baseline.json"


def test_wilson_matches_the_plan_examples() -> None:
    assert wilson(37, 41) == pytest.approx((0.774, 0.961), abs=0.002)
    assert wilson(0, 42) == pytest.approx((0.0, 0.084), abs=0.002)
    assert wilson(2, 42) == pytest.approx((0.013, 0.158), abs=0.002)
    assert wilson(0, 0) == (0.0, 1.0)


def test_percentile_nearest_rank() -> None:
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0
    assert percentile([1.0] * 99 + [50.0], 0.99) == 1.0
    assert percentile([], 0.5) == 0.0


def test_example_items_validate() -> None:
    items = load_items(ITEMS)
    assert len(items) >= 10
    assert {i.stage for i in items} == {"input", "output"}
    assert {i.label for i in items} == {"adversarial", "benign"}


def run(
    tmp_path: Path, *extra: str, items: Path = ITEMS, baseline: Path = BASELINE
) -> tuple[int, dict[str, Any]]:
    out = tmp_path / "result.json"
    code = main(
        [
            "--items",
            str(items),
            "--policies",
            str(POLICY_DIR),
            "--baseline",
            str(baseline),
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, json.loads(out.read_text()) if out.exists() else {}


def test_runner_reports_rates_with_cis_and_passes_against_the_baseline(tmp_path: Path) -> None:
    # the rule-only cli catches every example item; the full set needs the gate's jev recording and models
    examples = tmp_path / "items"
    examples.mkdir()
    for name in ("input-examples.yaml", "output-examples.yaml"):
        shutil.copy(ITEMS / name, examples / name)
    code, result = run(tmp_path, items=examples)
    assert code == 0
    assert result["schema_version"] == 1
    assert result["suite"] == "guardrails"
    assert result["status"] == "pass"
    metrics = result["metrics"]
    for name in ("catch_rate", "fpr", "input_catch_rate", "output_fpr"):
        assert {"value", "n", "k", "ci95"} <= set(metrics[name])
    assert metrics["catch_rate"]["value"] == 1.0
    assert metrics["fpr"]["k"] == 0
    assert all(i["outcome"] in ("caught", "allowed") for i in result["items"])
    assert result["policy"]["id"] == "default"
    output_items = [i for i in result["items"] if i["stage"] == "output"]
    assert all(i["leak_free"] and not i["stream_mismatch"] for i in output_items)


def test_item_regressions_fail_the_gate(tmp_path: Path) -> None:
    items = tmp_path / "items"
    items.mkdir()
    flipped = [
        {
            "id": "in-reg-001",
            "stage": "input",
            "category": "regression",
            "label": "adversarial",
            "messages": [{"role": "user", "content": "a perfectly harmless sentence"}],
            "expected": {"action": "block"},
            "added_in": "1.0.0",
        }
    ]
    (items / "input-regression.yaml").write_text(yaml.safe_dump(flipped))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"schema_version": 1, "items": {"in-reg-001": "caught"}}))
    code, result = run(tmp_path, items=items, baseline=baseline)
    assert code == 1
    assert result["regressions"] == [{"id": "in-reg-001", "from": "caught", "to": "missed"}]
    assert result["gates"][0]["status"] == "fail"


def test_update_baseline_and_schema_errors(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    code, _ = run(tmp_path, "--update-baseline", baseline=baseline)
    assert code == 0
    recorded = json.loads(baseline.read_text())
    assert recorded["items"]["in-ovr-001"] == "caught"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.yaml").write_text(yaml.safe_dump([{"id": "nope"}]))
    code, _ = run(tmp_path / "b", items=bad, baseline=baseline)
    assert code == 2
