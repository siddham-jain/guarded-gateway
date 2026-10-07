import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from gg.observability.metrics import Metrics

DEPLOY = Path(__file__).resolve().parents[3] / "deploy"
DASHBOARDS = sorted((DEPLOY / "grafana" / "dashboards").glob("*.json"))
ALERTS = DEPLOY / "prometheus" / "alerts.yml"
METRIC = re.compile(r"\bgg_[a-z0-9_]+")
OVERVIEW_ROWS = [
    "Headline",
    "Traffic & errors",
    "Latency",
    "Cost & tokens",
    "Cache",
    "Routing & fallbacks",
    "Breakers & rate limits",
    "Guardrails",
]


def exposed_names() -> set[str]:
    """every sample name /metrics can expose for the gg_* catalogue"""
    suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count"), "gauge": ("",)}
    names: set[str] = set()
    for family in Metrics(process_collectors=False).registry.collect():
        names.update(family.name + suffix for suffix in suffixes[family.type])
    return names


def panels(dashboard: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for panel in dashboard["panels"]:
        yield panel
        yield from panel.get("panels", ())


def queries(dashboard: dict[str, Any]) -> Iterator[str]:
    for panel in panels(dashboard):
        for target in panel.get("targets", ()):
            yield target["expr"]
    for variable in dashboard["templating"]["list"]:
        yield variable["definition"]


def balanced(expr: str) -> bool:
    depth = {"(": 0, "{": 0, "[": 0}
    closing = {")": "(", "}": "{", "]": "["}
    for ch in expr:
        if ch in depth:
            depth[ch] += 1
        elif ch in closing:
            depth[closing[ch]] -= 1
            if depth[closing[ch]] < 0:
                return False
    return not any(depth.values())


def test_dashboards_exist() -> None:
    assert {p.name for p in DASHBOARDS} >= {"gg-overview.json", "gg-providers.json"}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_queries_use_catalogue_metrics(path: Path) -> None:
    dashboard = json.loads(path.read_text())
    known = exposed_names()
    exprs = list(queries(dashboard))
    assert exprs
    for expr in exprs:
        assert balanced(expr), expr
        unknown = set(METRIC.findall(expr)) - known
        assert not unknown, f"{path.name}: {sorted(unknown)} in {expr}"


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.name)
def test_dashboard_panels_are_well_formed(path: Path) -> None:
    dashboard = json.loads(path.read_text())
    ids = [p["id"] for p in panels(dashboard)]
    assert len(ids) == len(set(ids))
    for panel in panels(dashboard):
        if panel["type"] == "row":
            continue
        assert panel["datasource"]["uid"] == "gg-prometheus", panel["title"]
        assert panel.get("description"), panel["title"]
        assert "instance=" not in json.dumps(panel["targets"]), panel["title"]


def test_overview_has_the_planned_rows() -> None:
    dashboard = json.loads((DEPLOY / "grafana" / "dashboards" / "gg-overview.json").read_text())
    rows = [p["title"] for p in dashboard["panels"] if p["type"] == "row"]
    assert rows == OVERVIEW_ROWS


def test_alert_rules_use_catalogue_metrics() -> None:
    rules = yaml.safe_load(ALERTS.read_text())
    known = exposed_names()
    alerts = [rule for group in rules["groups"] for rule in group["rules"]]
    assert len({rule["alert"] for rule in alerts}) == len(alerts)
    for rule in alerts:
        assert balanced(rule["expr"]), rule["alert"]
        assert set(METRIC.findall(rule["expr"])) <= known, rule["alert"]
        assert rule["annotations"]["summary"]
