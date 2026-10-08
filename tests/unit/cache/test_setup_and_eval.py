from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis
import pytest

from gg.cache.base import CONSUMED_EXTENSIONS, Embedder
from gg.cache.config import CacheConfig, EmbedderConfig
from gg.cache.embedders import build_embedder
from gg.cache.embedders.fastembed import FastEmbedEmbedder
from gg.cache.embedders.hashing import HashingEmbedder
from gg.cache.eval.__main__ import main as eval_main
from gg.cache.eval.pairs import Pair, load_pairs
from gg.cache.eval.sweep import choose_tau, confusion, score_pairs, sweep, thresholds
from gg.cache.recorder import StreamRecorder
from gg.cache.setup import build_cache
from gg.config.loader import load_file
from gg.core.aio import CpuExecutor, TaskSupervisor
from gg.core.clock import FakeClock
from gg.core.schema import ChatChunk
from tests.unit.cache.support import RecordingHooks, Upstream, chunks, config, ctx_for, run

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "evals" / "cache" / "pairs.jsonl"


def test_repo_config_pairs_the_wide_threshold_with_the_verifier() -> None:
    cfg = load_file(ROOT / "config" / "cache.yaml", CacheConfig)
    assert cfg.semantic.distance_threshold == pytest.approx(0.15)
    assert cfg.semantic.verifier.type == "jev"
    assert cfg.semantic.embedder.dim == 384


@pytest.mark.parametrize(
    "bad",
    [
        {"exact": {"min_ttl_s": 900, "max_ttl_s": 60}},
        {"semantic": {"min_user_chars": 50, "max_user_chars": 10}},
        {"exact": {"typo": 1}},
    ],
)
def test_invalid_config_fails(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        CacheConfig.model_validate(bad)


async def test_build_without_redis_uses_memory_backends() -> None:
    cache = build_cache(
        config(), redis=None, clock=FakeClock(), supervisor=TaskSupervisor(), embedder=HashingEmbedder(64)
    )
    await cache.start()
    assert cache.semantic_probe is not None
    assert cache.consumed_extensions == CONSUMED_EXTENSIONS
    assert cache.exact_stage.name == "cache_exact"
    assert cache.semantic_probe.precedence == 50
    await cache.stop()


def test_hash_tracks_config_and_embedder() -> None:
    def digest(cfg: CacheConfig, embedder: Embedder | None) -> str:
        return build_cache(
            cfg, redis=None, clock=FakeClock(), supervisor=TaskSupervisor(), embedder=embedder
        ).hash

    base = digest(config(), HashingEmbedder(64))
    assert base == digest(config(), HashingEmbedder(64))
    assert base != digest(config(), HashingEmbedder(32))
    assert base != digest(config(max_ttl_s=3600), HashingEmbedder(64))


async def test_build_with_redis_without_query_engine_keeps_exact_only() -> None:
    redis = fakeredis.FakeAsyncRedis()
    hooks = RecordingHooks()
    cache = build_cache(
        config(),
        redis=redis,
        clock=FakeClock(),
        supervisor=TaskSupervisor(),
        embedder=HashingEmbedder(64),
        metrics_hooks=hooks,
    )
    await cache.start()
    assert cache.vector_index is not None
    assert not cache.vector_index.available
    upstream = Upstream()
    semantic_key = {"cache": {"semantic": True}}
    prompt = [{"role": "user", "content": "What is the capital of France?"}]
    await run(cache, ctx_for(key=semantic_key, messages=prompt), upstream)
    hit = ctx_for(key=semantic_key, messages=prompt)
    await run(cache, hit, upstream)
    assert hit.cache_status == "exact_hit"
    assert upstream.calls == 1
    assert ("semantic", "stored", "ok") not in hooks.stores
    assert [k async for k in redis.scan_iter("gg:c:v1:*")]
    await redis.aclose()


def test_missing_fastembed_disables_the_semantic_layer() -> None:
    cpu = CpuExecutor(workers=1, queue_max=1)
    try:
        embedder = build_embedder(EmbedderConfig(), cpu)
        assert isinstance(embedder, FastEmbedEmbedder)
        assert embedder.name == "bge-small-en-v1.5"
        cache = build_cache(
            config(), redis=None, clock=FakeClock(), supervisor=TaskSupervisor(), embedder=embedder
        )
        if not FastEmbedEmbedder.installed():
            assert cache.semantic_probe is None
    finally:
        cpu.shutdown()


async def test_hashing_embedder_is_deterministic_and_unit_length() -> None:
    embedder = HashingEmbedder(64)
    [a, b] = await embedder.embed(["Hello world", "Hello world"])
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert embedder.name == "hashing-64"
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(4)


async def test_recorder_overflow_and_completion() -> None:
    async def source() -> AsyncIterator[ChatChunk]:
        for c in chunks(["a" * 30, "b" * 30]):
            yield c

    small = StreamRecorder(max_chars=40)
    _ = [c async for c in small.tee(source())]
    assert small.complete
    assert small.overflow
    assert small.response() is None
    roomy = StreamRecorder(max_chars=1000)
    _ = [c async for c in roomy.tee(source())]
    reply = roomy.response()
    assert reply is not None
    assert reply.choices[0].message.content == "a" * 30 + "b" * 30
    assert reply.usage is not None


def test_repo_pairs_validate() -> None:
    pairs = load_pairs(PAIRS)
    assert len(pairs) >= 30
    assert {p.split for p in pairs} == {"dev", "test"}
    assert any(p.should_hit for p in pairs)
    assert any(not p.should_hit for p in pairs)


def test_pair_loader_rejects_duplicates_and_bad_rows(tmp_path: Path) -> None:
    row = '{"id": "p0001", "anchor": "a", "candidate": "b", "should_hit": true, "category": "paraphrase"}'
    dup = tmp_path / "dup.jsonl"
    dup.write_text(f"{row}\n\n{row}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_pairs(dup)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "p0001", "anchor": "a"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_pairs(bad)


def _pair(pid: str, a: str, b: str, hit: bool, split: str = "dev") -> Pair:
    return Pair.model_validate(
        {"id": pid, "anchor": a, "candidate": b, "should_hit": hit, "category": "paraphrase", "split": split}
    )


async def test_sweep_counts_and_chooses_tau() -> None:
    pairs = [
        _pair("p0001", "what is the capital of france", "What is the capital of France?", True),
        _pair("p0002", "what is 17 times 23", "what is 17 times 24", False),
        _pair("p0003", "explain dns", "explain tcp handshakes in depth", False, "test"),
    ]
    embedder = HashingEmbedder(64)
    plain = await score_pairs(pairs, embedder, num_sig=False)
    tagged = await score_pairs(pairs, embedder, num_sig=True)
    assert plain[0].distance == pytest.approx(0.0)
    assert plain[1].distance is not None
    assert tagged[1].distance is None
    row = confusion(tagged, 0.05)
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (1, 0, 0, 2)
    assert row["precision"] == 1.0
    assert choose_tau([confusion(tagged, t) for t in (0.0, 0.1)], 0.98) == 0.1
    assert choose_tau([{"tau": 0.1, "precision": None}], 0.98) is None
    result = await sweep(pairs, embedder, taus=thresholds(stop=0.1, step=0.05))
    assert set(result["variants"]) == {"embedding", "embedding+num_sig"}
    assert result["variants"]["embedding+num_sig"]["tau_star"] == 0.1
    assert len(result["variants"]["embedding"]["sweep"]) == 3


def test_thresholds_grid() -> None:
    grid = thresholds()
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(0.30)
    assert len(grid) == 61


def test_eval_cli_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sweep.json"
    assert eval_main(["--pairs", str(PAIRS), "--out", str(out)]) == 0
    assert out.is_file()
    assert "cache sweep" in capsys.readouterr().out
    assert eval_main(["--pairs", str(tmp_path / "missing.jsonl")]) == 2
