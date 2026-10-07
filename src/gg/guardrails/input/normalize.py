"""tier 0: strip invisible characters from forwarded text and build the views the detectors read"""

import base64
import binascii
import re
import unicodedata
from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field

from gg.core.guard_types import Verdict
from gg.core.schema import StrictModel
from gg.guardrails.base import Decoded, GuardContext, GuardFinding, GuardStage, Segment, Streaming, finding
from gg.guardrails.registry import GuardDeps

_ZERO_WIDTH = {cp: None for cp in (*range(0x200B, 0x2010), 0x00AD, *range(0x2060, 0x2065), 0xFEFF)}
_TAG_FIRST, _TAG_LAST = 0xE0000, 0xE007F
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{24,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){12,}(?![0-9A-Fa-f])")
_MAX_BLOBS = 16
_MAX_BLOB_CHARS = 16_384

# cyrillic and greek letters that render like latin ones; enough to defeat homoglyph-swapped keywords
_CONFUSABLES = str.maketrans(
    "аеорсухіјѕԁɡАВЕКМНОРСТХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοαιкνρτυχ",
    "aeopcyxijsdgABEKMHOPCTXIJSABEZHIKMNOPTYXoaikvptux",
)


class DecodeCfg(StrictModel):
    base64: bool = True
    hex: bool = True
    min_len: Annotated[int, Field(ge=8)] = 24
    max_depth: Annotated[int, Field(ge=1, le=4)] = 2
    min_printable: Annotated[float, Field(ge=0, le=1)] = 0.9


class NormalizeCfg(StrictModel):
    strip_invisible: bool = True
    inspect_nfkc: bool = True
    fold_confusables: bool = True
    decode: DecodeCfg = DecodeCfg()
    flag_invisible_ratio: Annotated[float, Field(ge=0, le=1)] = 0.02
    max_scan_chars: Annotated[int, Field(ge=1_000)] = 32_000
    on_oversize: Literal["flag", "block"] = "flag"


def _printable_text(raw: bytes, min_printable: float) -> str | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return None
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return text if printable / len(text) >= min_printable else None


def _decode_blob(blob: str, cfg: DecodeCfg) -> str | None:
    if cfg.hex and _HEX_RE.fullmatch(blob):
        try:
            return _printable_text(bytes.fromhex(blob), cfg.min_printable)
        except ValueError:
            return None
    if cfg.base64:
        padded = blob + "=" * (-len(blob) % 4)
        try:
            raw = base64.b64decode(padded.replace("-", "+").replace("_", "/"), validate=True)
        except (binascii.Error, ValueError):
            return None
        return _printable_text(raw, cfg.min_printable)
    return None


def find_encoded(text: str, cfg: DecodeCfg) -> list[Decoded]:
    """printable payloads hidden in base64/hex blobs, decoded up to max_depth levels"""
    out: list[Decoded] = []
    for m in (*_HEX_RE.finditer(text), *_BASE64_RE.finditer(text)):
        if len(out) >= _MAX_BLOBS:
            break
        blob = m.group(0)
        if len(blob) < cfg.min_len or len(blob) > _MAX_BLOB_CHARS or any(d.start == m.start() for d in out):
            continue
        payload = _decode_blob(blob, cfg)
        depth = 1
        while payload is not None:
            out.append(Decoded(m.start(), m.end(), payload))
            inner = _BASE64_RE.fullmatch(payload.strip()) or _HEX_RE.fullmatch(payload.strip())
            if depth >= cfg.max_depth or inner is None:
                break
            payload = _decode_blob(inner.group(0), cfg)
            depth += 1
    return out


class Normalize:
    name: str = "normalize"
    stage: GuardStage = "input"
    tier: int = 0
    streaming: Streaming = "windowed"

    def __init__(self, cfg: NormalizeCfg) -> None:
        self._cfg = cfg

    def _inspect_view(self, text: str) -> str:
        cfg = self._cfg
        view = unicodedata.normalize("NFKC", text) if cfg.inspect_nfkc else text
        if cfg.fold_confusables:
            view = view.translate(_CONFUSABLES)
        if len(view) > cfg.max_scan_chars:
            half = cfg.max_scan_chars // 2
            view = f"{view[:half]}\n{view[-half:]}"
        return view

    def normalize(self, seg: Segment) -> tuple[Segment, set[str]]:
        cfg = self._cfg
        labels: set[str] = set()
        # unicode tag characters smuggle invisible ascii; decode it so the rules can read it
        smuggled = "".join(chr(ord(ch) - _TAG_FIRST) for ch in seg.text if 0xE0020 <= ord(ch) <= 0xE007E)
        if smuggled:
            labels.add("tag_chars")
        stripped = seg.text.translate(_ZERO_WIDTH)
        stripped = "".join(ch for ch in stripped if not _TAG_FIRST <= ord(ch) <= _TAG_LAST)
        removed = len(seg.text) - len(stripped)
        if seg.text and removed / len(seg.text) >= cfg.flag_invisible_ratio:
            labels.add("invisible")
        text = stripped if cfg.strip_invisible else seg.text
        if len(text) > cfg.max_scan_chars:
            labels.add("oversize")
        decoded = find_encoded(text, cfg.decode)
        if smuggled:
            decoded.append(Decoded(0, 0, smuggled))
        out = replace(seg, text=text, inspect=self._inspect_view(stripped), decoded=tuple(decoded))
        return out, labels

    async def check(self, gctx: GuardContext, /) -> GuardFinding:
        replacements: list[Segment] = []
        labels: set[str] = set()
        for seg in gctx.segments:
            new, seg_labels = self.normalize(seg)
            replacements.append(new)
            labels |= seg_labels
        verdict = Verdict.FLAG if labels else Verdict.ALLOW
        if "oversize" in labels and self._cfg.on_oversize == "block":
            verdict = Verdict.BLOCK
        return finding(
            self.name,
            "input",
            verdict,
            reason=",".join(sorted(labels)),
            labels=tuple(sorted(labels)),
            replacements=tuple(replacements),
        )


def create(cfg: NormalizeCfg, deps: GuardDeps) -> Normalize:
    return Normalize(cfg)
