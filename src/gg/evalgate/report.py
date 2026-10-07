"""markdown summary for $GITHUB_STEP_SUMMARY (C11 §3.11 comment shape)"""

from typing import Any

from gg.guardrails.eval.stats import wilson

HEADLINES = {
    "guardrails": ("input_catch_rate", "input_fpr", "output_catch_rate", "output_fpr"),
    "cache": ("precision", "hit_rate"),
}


def overall(results: dict[str, dict[str, Any]]) -> str:
    return "fail" if any(r["status"] == "fail" for r in results.values()) else "pass"


def _metric(name: str, metric: dict[str, Any] | None) -> str:
    if metric is None or metric.get("value") is None:
        return f"{name} n/a"
    if "k" not in metric:
        return f"{name} {metric['value']:g}"
    low, high = wilson(metric["k"], metric["n"])
    return f"{name} {metric['k']}/{metric['n']} = {metric['value']:.3f} [{low:.2f}, {high:.2f}]"


def _headline(suite: str, result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return " · ".join(_metric(name, metrics.get(name)) for name in HEADLINES.get(suite, ()))


def render(results: dict[str, dict[str, Any]], *, base_ref: str | None) -> str:
    baseline = f"baselines and accepted changes from `{base_ref}`" if base_ref else "working-tree baselines"
    lines = [
        f"### GG eval gate: {overall(results).upper()}",
        "",
        f"Replay mode, no network, {baseline}.",
        "",
        "| suite | status | headline | covers |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {suite} | {r['status']} | {_headline(suite, r)} | {r.get('coverage', '')} |"
        for suite, r in results.items()
    ]
    for suite, r in results.items():
        lines += ["", f"#### {suite}", "", "| gate | mode | status | detail |", "|---|---|---|---|"]
        lines += [f"| {g['name']} | {g['mode']} | {g['status']} | {g['detail']} |" for g in r["gates"]]
    regressions = [
        (suite, reg, {i["id"]: i for i in r["items"]}.get(reg["id"], {}))
        for suite, r in results.items()
        for reg in r["regressions"]
    ]
    if regressions:
        lines += [
            "",
            "#### Item regressions (not in accepted_changes)",
            "",
            "| suite | item | category | baseline | now |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {suite} | {reg['id']} | {item.get('category', '')} | {reg['from']} | {reg['to']} |"
            for suite, reg, item in regressions
        ]
        lines += [
            "",
            "To accept a change on purpose, add an entry with a reason to "
            "`evals/<suite>/accepted_changes.yaml` in the same change.",
        ]
    return "\n".join(lines) + "\n"
