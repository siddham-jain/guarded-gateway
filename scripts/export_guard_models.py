"""dev-only: export guard classifiers to dynamic-int8 onnx under .models/<id>/ (never baked into an image).

needs packages the runtime never installs:
    uv run --with "optimum[onnxruntime]>=1.24" --with torch --with transformers \
        python scripts/export_guard_models.py granite-guardian-hap-38m-int8
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# target id in gg.guardrails.ml.artefacts -> (source repo, pinned revision, licence notice)
EXPORTS: dict[str, tuple[str, str, str]] = {
    "granite-guardian-hap-38m-int8": (
        "ibm-granite/granite-guardian-hap-38m",
        "faf9291163c7363d4a7ba7fc1dc72243214af931",
        "ibm-granite/granite-guardian-hap-38m, Apache-2.0.",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(model_id: str, out_root: Path) -> Path:
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    repo, revision, notice = EXPORTS[model_id]
    out = out_root / model_id
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model = ORTModelForSequenceClassification.from_pretrained(repo, revision=revision, export=True)
        model.save_pretrained(tmp)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        tokenizer.save_pretrained(tmp)
        # Gather too: the embedding table is most of a small classifier's weights
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        qconfig.operators_to_quantize = [*qconfig.operators_to_quantize, "Gather"]
        ORTQuantizer.from_pretrained(tmp).quantize(save_dir=tmp, quantization_config=qconfig)
        shutil.copy(Path(tmp) / "model_quantized.onnx", out / "model.int8.onnx")
        shutil.copy(Path(tmp) / "tokenizer.json", out / "tokenizer.json")
        shutil.copy(Path(tmp) / "config.json", out / "config.json")
    (out / "NOTICE").write_text(notice + "\n")
    manifest = {
        "source": repo,
        "revision": revision,
        "quant": "dynamic-int8",
        "files": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in (".onnx", ".json")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+", choices=sorted(EXPORTS))
    parser.add_argument("--out", type=Path, default=ROOT / ".models")
    args = parser.parse_args()
    for model_id in args.models:
        print(f"exported {model_id} -> {export(model_id, args.out)}")


if __name__ == "__main__":
    main()
