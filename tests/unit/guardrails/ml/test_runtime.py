"""model runtime: single load, readiness, artefact store, builder wiring, lazy ml imports"""

import asyncio
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gg.config.loader import ConfigError
from gg.core.aio import CpuExecutor
from gg.core.clock import SystemClock
from gg.guardrails.ml.artefacts import MODELS, ArtefactError, ArtefactStore
from gg.guardrails.ml.runtime import MlRuntime, Resource
from gg.guardrails.setup import build_guardrails
from tests.unit.guardrails.ml.support import FakeEmbedder
from tests.unit.guardrails.support import POLICY_DIR, ROOT


@pytest.fixture
async def cpu() -> AsyncIterator[CpuExecutor]:
    executor = CpuExecutor(2, 8)
    yield executor
    executor.shutdown()


async def test_resource_loads_once_even_when_a_waiter_times_out() -> None:
    loads = 0
    release = asyncio.Event()

    async def load() -> str:
        nonlocal loads
        loads += 1
        await release.wait()
        return "model"

    res = Resource("m", load)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await res.get()
    waiter = asyncio.create_task(res.get())
    await asyncio.sleep(0)
    release.set()
    assert await waiter == "model"
    assert await res.get() == "model"
    assert loads == 1
    assert res.loaded
    assert res.load_s is not None


async def test_health_is_down_until_started_then_reports_failures(tmp_path: Path, cpu: CpuExecutor) -> None:
    runtime = MlRuntime(cpu, ArtefactStore(tmp_path, download=False))

    async def ok() -> int:
        return 1

    async def broken() -> int:
        raise RuntimeError("no weights")

    runtime.resource("ok", ok)
    runtime.resource("lazy-broken", broken, lazy=True)
    assert (await runtime.check()).status == "down"
    await runtime.start()
    status = await runtime.check()
    assert status.status == "ok"
    runtime.resource("broken", broken)
    await runtime.start()
    status = await runtime.check()
    assert status.status == "down"
    assert "broken" in status.detail
    assert "lazy-broken" not in status.detail
    assert {r.name: r.loaded for r in runtime.report()} == {"ok": True, "lazy-broken": False, "broken": False}


async def test_missing_artefacts_fail_with_a_hint(tmp_path: Path) -> None:
    store = ArtefactStore(tmp_path, download=False)
    with pytest.raises(ArtefactError, match="export_guard_models"):
        store.fetch(MODELS["granite-guardian-hap-38m-int8"])
    with pytest.raises(ArtefactError, match="downloads are disabled"):
        store.fetch(MODELS["granite-guardian-hap-38m"])
    assert not store.present(MODELS["granite-guardian-hap-38m"])


async def test_builder_wires_probe_and_shares_one_model_per_id(cpu: CpuExecutor, tmp_path: Path) -> None:
    guardrails = build_guardrails(
        POLICY_DIR,
        clock=SystemClock(),
        cpu=cpu,
        embedder=FakeEmbedder(["dose", "gun"]),
        models_dir=tmp_path,
        download_models=False,
    )
    assert guardrails.tier2_probe is not None
    assert guardrails.tier2_probe.precedence == 10
    assert guardrails.ml is not None
    names = sorted(r.name for r in guardrails.ml.report())
    # every override combination was built at startup, yet each model is registered once
    assert names == [
        "model:granite-guardian-hap-38m",
        "presidio:en_core_web_sm",
        names[2],
    ]
    assert names[2].startswith("topic:fake-embedder:")
    assert (await guardrails.health.check()).status == "down"


async def test_builder_without_models_dir_disables_ml(cpu: CpuExecutor) -> None:
    guardrails = build_guardrails(POLICY_DIR, clock=SystemClock(), cpu=cpu)
    assert guardrails.tier2_probe is None
    assert guardrails.ml is None
    await guardrails.start()
    assert (await guardrails.health.check()).status == "ok"
    with pytest.raises(ValueError, match="cpu executor"):
        build_guardrails(POLICY_DIR, clock=SystemClock(), models_dir=Path(".models"))


def test_topic_without_the_shared_embedder_is_a_config_error(cpu: CpuExecutor, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="shared embedder"):
        build_guardrails(POLICY_DIR, clock=SystemClock(), cpu=cpu, models_dir=tmp_path)


def test_core_app_imports_without_loading_ml_libraries() -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "import gg.app.factory\n"
        "from gg.core.aio import CpuExecutor\n"
        "from gg.core.clock import SystemClock\n"
        "from gg.guardrails.setup import build_guardrails\n"
        "from gg.cache.embedders.hashing import HashingEmbedder\n"
        f"build_guardrails(Path({str(POLICY_DIR)!r}), clock=SystemClock(), cpu=CpuExecutor(1, 1),\n"
        "                 embedder=HashingEmbedder(8), models_dir=Path('unused'), download_models=False)\n"
        "heavy = {'onnxruntime', 'numpy', 'tokenizers', 'presidio_analyzer', 'spacy', 'fastembed',\n"
        "         'huggingface_hub'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
