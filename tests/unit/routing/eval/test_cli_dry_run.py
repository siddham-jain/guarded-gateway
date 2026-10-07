"""the whole harness through the real composition root: ci profile, in-process mock provider, fake scorer"""

import json
from pathlib import Path

import pytest

from gg.cli import routing_eval
from gg.routing.eval.budget import Price

ROOT = Path(__file__).resolve().parents[4]
BASE = ["--suite", str(ROOT / "evals/routing/suite.yaml"), "--config-dir", str(ROOT / "config")]


def run(tmp_path: Path, *extra: str) -> int:
    return routing_eval.main([*BASE, "--dry-run", "--out", str(tmp_path), "--bootstrap", "50", *extra])


def test_dry_run_end_to_end_then_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "# Routing eval - dry-run" in out
    for name in ("result.json", "report.md", "curves.csv", "cost_quality.svg"):
        assert (tmp_path / name).is_file()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["mode"] == "dry-run"
    assert result["dataset"]["evaluated"] == 170
    assert result["missing"] == {}
    routers = result["metrics"]["all"]["routers"]
    assert routers["random"]["apgr"] == 0.5
    assert routers["scorer"]["apgr"] > routers["random"]["apgr"]
    assert result["cost"]["phases"]["generate"]["called"] == 340
    assert (tmp_path / "cost_quality.svg").read_text().startswith("<svg")

    assert run(tmp_path) == 0
    again = json.loads((tmp_path / "result.json").read_text())
    assert again["cost"]["phases"]["generate"] == {"cached": 340, "called": 0, "failed": 0, "errors": []}
    assert again["metrics"]["all"]["routers"]["scorer"] == routers["scorer"]


def test_offline_report_reads_the_stores(tmp_path: Path) -> None:
    assert run(tmp_path, "--limit", "13") == 0
    assert run(tmp_path, "--limit", "13", "--offline") == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["mode"] == "offline"
    assert result["scorer_version"] == "fake-v1"
    assert result["dataset"]["evaluated"] == 13


def test_refuses_above_max_usd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(routing_eval, "_prices", lambda catalog, pair: {"mock/echo": Price(500, 500)})
    assert run(tmp_path, "--max-usd", "0.10") == 3
    assert "refusing" in capsys.readouterr().err
    assert not (tmp_path / "generations.jsonl").exists()


def test_estimate_only_makes_no_calls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, "--estimate-only") == 0
    assert "estimate: 372 uncached calls" in capsys.readouterr().out
    assert not (tmp_path / "scores.jsonl").exists()


def test_config_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert routing_eval.main([*BASE, "--pair", "nope", "--out", str(tmp_path)]) == 2
    assert run(tmp_path, "--pair", "dev-free") == 2
    assert "error:" in capsys.readouterr().err
