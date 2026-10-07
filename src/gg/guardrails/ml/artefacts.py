"""guard model artefacts: pinned sources per file and a local store under .models/<id>/"""

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type FileRole = Literal["onnx", "tokenizer", "config"]


@dataclass(frozen=True, slots=True)
class Source:
    """one file of an artefact; repo None means it only exists locally (exported by scripts/)"""

    path: str
    repo: str | None = None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    licence: str
    files: Mapping[FileRole, Source]
    # index of the "unsafe" class in the logits
    positive: int = 1
    max_length: int = 512
    stride: int = 64
    notes: str = ""


_GRANITE = "ibm-granite/granite-guardian-hap-38m"
_GRANITE_REV = "faf9291163c7363d4a7ba7fc1dc72243214af931"
_GRANITE_ONNX = "KantiArumilli/granite-guardian-hap-38m-onnx"
_GRANITE_ONNX_REV = "06e1ea287b08a2cab49876a647a08a2e5c11712b"
_HHEM = "FamiliarTools/HHEM-2.1-Open-onnx"
_HHEM_REV = "837e48491cbfe5a7334b26b6e2e9d7b2ff074b12"

MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="granite-guardian-hap-38m",
            licence="apache-2.0",
            files={
                "onnx": Source("guardian_model.onnx", _GRANITE_ONNX, _GRANITE_ONNX_REV),
                "tokenizer": Source("tokenizer.json", _GRANITE, _GRANITE_REV),
                "config": Source("config.json", _GRANITE, _GRANITE_REV),
            },
            notes="ibm tokenizer/config; fp32 onnx from a third-party export of the same weights",
        ),
        ModelSpec(
            id="granite-guardian-hap-38m-int8",
            licence="apache-2.0",
            files={
                "onnx": Source("model.int8.onnx"),
                "tokenizer": Source("tokenizer.json"),
                "config": Source("config.json"),
            },
            notes="ibm-granite/granite-guardian-hap-38m exported by scripts/export_guard_models.py",
        ),
        ModelSpec(
            id="hhem-2.1-open",
            licence="apache-2.0",
            files={
                "onnx": Source("model.onnx", _HHEM, _HHEM_REV),
                "tokenizer": Source("tokenizer.json", _HHEM, _HHEM_REV),
            },
            max_length=2048,
            notes="vectara hhem-2.1-open (flan-t5-base); third-party onnx export, parity-checked upstream",
        ),
    )
}


class ArtefactError(Exception):
    pass


def model_spec(model_id: str) -> ModelSpec:
    spec = MODELS.get(model_id)
    if spec is None:
        raise ValueError(f"unknown guard model '{model_id}'; known: {', '.join(sorted(MODELS))}")
    return spec


class ArtefactStore:
    """files live at <root>/<model id>/<path>; missing pinned files are fetched from hugging face"""

    def __init__(self, root: Path, *, download: bool = True) -> None:
        self.root = root
        self._download = download

    def local(self, spec: ModelSpec) -> dict[FileRole, Path]:
        return {role: self.root / spec.id / src.path for role, src in spec.files.items()}

    def present(self, spec: ModelSpec) -> bool:
        return all(p.is_file() for p in self.local(spec).values())

    def fetch(self, spec: ModelSpec) -> dict[FileRole, Path]:
        """blocking; runs on the cpu executor at startup"""
        paths = self.local(spec)
        for role, src in spec.files.items():
            if paths[role].is_file():
                continue
            if src.repo is None:
                raise ArtefactError(f"{spec.id}: {paths[role]} is missing; {spec.notes}")
            if not self._download:
                raise ArtefactError(f"{spec.id}: {paths[role]} is missing and downloads are disabled")
            self._hf_download(spec, src)
        return paths

    def _hf_download(self, spec: ModelSpec, src: Source) -> None:
        try:
            hub = importlib.import_module("huggingface_hub")
        except ImportError as exc:
            raise ArtefactError("downloading guard models needs the 'ml' extra") from exc
        # ungated sources only: never send a token from the user's global hugging face login
        hub.hf_hub_download(
            src.repo, src.path, revision=src.revision, local_dir=self.root / spec.id, token=False
        )
