import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gg.evalgate.__main__ import main
from gg.evalgate.refs import GitError, read_at_ref

ROOT = Path(__file__).resolve().parents[3]

ACCEPT_P0016 = """- item: p0016
  from: rejected
  to: false_hit
  reason: number_change pair with identical digits, known hashing collision
"""


@pytest.fixture(autouse=True)
def _no_step_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    # under ci the runner's real summary file must not collect test output
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def _copy_evals(dest: Path) -> Path:
    evals = dest / "evals"
    shutil.copytree(ROOT / "evals", evals, ignore=shutil.ignore_patterns("results"))
    return evals


STANDIN_SUITE = """suite: cache
version: 1.0.0
embedder: {provider: hashing, name: hashing, dim: 256}
threshold: 0.135
gates:
  item_regressions: {mode: gate}
"""


def _tamper_cache_baseline(evals: Path) -> None:
    # without a recording the gate scores pairs with the hashing stand-in, where p0016 is a false hit;
    # a baseline that says rejected makes it a regression
    shutil.rmtree(evals / "cache" / "cassettes")
    (evals / "cache" / "suite.yaml").write_text(STANDIN_SUITE)
    assert _run("--suite", "cache", "--evals", evals, "--update-baselines") == 0
    path = evals / "cache" / "baseline.json"
    baseline = json.loads(path.read_text())
    assert baseline["items"]["p0016"] == "false_hit"
    baseline["items"]["p0016"] = "rejected"
    path.write_text(json.dumps(baseline))


def _run(*args: str | Path) -> int:
    return main([str(a) for a in args])


def test_committed_baselines_pass(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n")
    code = _run(
        "--evals", ROOT / "evals", "--config", ROOT / "config", "--out-dir", out, "--summary", summary
    )
    assert code == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["status"] == "pass"
    assert set(gate["suites"]) == {"guardrails", "cache"}
    assert json.loads((out / "guardrails.json").read_text())["dataset"]["n_items"] > 0
    assert json.loads((out / "cache.json").read_text())["threshold"] > 0
    text = summary.read_text()
    assert text.startswith("earlier step\n### GG eval gate: PASS")
    assert (out / "summary.md").read_text() in text


def test_cache_regression_fails_until_accepted(tmp_path: Path) -> None:
    evals = _copy_evals(tmp_path)
    _tamper_cache_baseline(evals)
    out = tmp_path / "out"
    assert _run("--suite", "cache", "--evals", evals, "--out-dir", out) == 1
    gate = json.loads((out / "gate.json").read_text())
    assert gate["suites"]["cache"]["regressions"] == [{"id": "p0016", "from": "rejected", "to": "false_hit"}]
    assert "| cache | p0016 | number_change | rejected | false_hit |" in (out / "summary.md").read_text()

    (evals / "cache" / "accepted_changes.yaml").write_text(ACCEPT_P0016)
    assert _run("--suite", "cache", "--evals", evals, "--out-dir", out) == 0
    gate = json.loads((out / "gate.json").read_text())
    assert gate["suites"]["cache"]["accepted_regressions"][0]["id"] == "p0016"


def test_staged_rule_narrowing_is_blocked(tmp_path: Path) -> None:
    # the docs/ci.md demonstration change: anchor the classic override rule to the start of a line
    config = tmp_path / "config"
    shutil.copytree(ROOT / "config", config)
    rules = config / "policies" / "rules" / "injection.v1.yaml"
    old = "pattern: '(?i)(?<!not )(?<!n''t )(?<!never )\\b(?:ignore|disregard"
    new = "pattern: '(?im)^\\s*(?:please\\s+)?(?:ignore|disregard"
    text = rules.read_text()
    assert old in text
    rules.write_text(text.replace(old, new, 1))
    out = tmp_path / "out"
    assert _run("--suite", "guardrails", "--evals", ROOT / "evals", "--config", config, "--out-dir", out) == 1
    gate = json.loads((out / "gate.json").read_text())["suites"]["guardrails"]
    assert gate["regressions"]
    assert {g["name"]: g["status"] for g in gate["gates"]}["item_regressions"] == "fail"


def test_update_baselines_rewrites_them(tmp_path: Path) -> None:
    evals = _copy_evals(tmp_path)
    _tamper_cache_baseline(evals)
    assert _run("--suite", "cache", "--evals", evals, "--update-baselines") == 0
    assert json.loads((evals / "cache" / "baseline.json").read_text())["items"]["p0016"] == "false_hit"
    assert _run("--suite", "cache", "--evals", evals, "--out-dir", tmp_path / "out") == 0


def test_schema_errors_exit_2(tmp_path: Path) -> None:
    evals = _copy_evals(tmp_path)
    (evals / "cache" / "suite.yaml").write_text("suite: cache\nversion: 1.0.0\ngates: {}\n")
    assert _run("--suite", "cache", "--evals", evals, "--out-dir", tmp_path / "out") == 2
    (evals / "guardrails" / "suite.yaml").write_text("suite: cache\nversion: 1.0.0\ngates: {}\n")
    assert _run("--suite", "guardrails", "--evals", evals, "--out-dir", tmp_path / "out") == 2


def _git(repo: Path, *args: str) -> None:
    git = shutil.which("git")
    assert git is not None
    subprocess.run(  # noqa: S603
        [git, "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_base_ref_reads_baselines_and_only_new_accepted_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    evals = _copy_evals(repo)
    _tamper_cache_baseline(evals)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    monkeypatch.chdir(repo)
    cache = ("--suite", "cache", "--out-dir", tmp_path / "out")

    assert _run(*cache, "--base-ref", "HEAD") == 1
    accepted = Path("evals/cache/accepted_changes.yaml")
    accepted.write_text(ACCEPT_P0016)
    assert _run(*cache, "--base-ref", "HEAD") == 0

    _git(repo, "commit", "-q", "-am", "accept")
    assert _run(*cache, "--base-ref", "HEAD") == 1, "an entry already on the base excuses nothing"
    assert _run(*cache, "--base-ref", "HEAD~1") == 0

    assert _run(*cache, "--update-baselines") == 0
    assert _run(*cache, "--base-ref", "HEAD") == 1, "a working-tree baseline edit can't hide a regression"
    assert _run(*cache) == 0

    assert _run(*cache, "--base-ref", "no-such-ref") == 2
    assert read_at_ref("HEAD", Path("evals/cache/missing.json")) is None
    with pytest.raises(GitError):
        read_at_ref("no-such-ref", Path("evals/cache/baseline.json"))


def test_loosening_the_cache_verifier_is_blocked(tmp_path: Path) -> None:
    # the recording holds jev's scores, so a lower min_score replays as wrong answers being served
    config = tmp_path / "config"
    shutil.copytree(ROOT / "config", config)
    cache = config / "cache.yaml"
    text = cache.read_text()
    assert "min_score: 0.8" in text
    cache.write_text(text.replace("min_score: 0.8", "min_score: 0.0", 1))
    out = tmp_path / "out"
    assert _run("--suite", "cache", "--evals", ROOT / "evals", "--config", config, "--out-dir", out) == 1
    gate = json.loads((out / "gate.json").read_text())["suites"]["cache"]
    statuses = {g["name"]: g["status"] for g in gate["gates"]}
    assert (statuses["item_regressions"], statuses["precision"]) == ("fail", "fail")
    assert {"id": "p0016", "from": "rejected", "to": "false_hit"} in gate["regressions"]


def test_live_mode_refuses_to_write_baselines(tmp_path: Path) -> None:
    assert _run("--live", "--update-baselines", "--out-dir", tmp_path / "out") == 2
