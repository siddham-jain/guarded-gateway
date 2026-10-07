from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

from gg.guardrails.base import Redaction, Segment
from gg.guardrails.vault import GuardVault

# overlap priority: secrets beat financial ids beat contact data beat names
_PRIORITY: dict[str, int] = {"SECRET": 4, "CARD": 3, "IBAN": 3, "SSN": 3, "EMAIL": 2, "PHONE": 2, "IP": 2}

MARKER = "[REDACTED:{label}]"


def merge(spans: Iterable[Redaction]) -> list[Redaction]:
    """non-overlapping spans sorted by start; higher priority, then longer, wins an overlap"""
    ranked = sorted(spans, key=lambda s: (-_PRIORITY.get(s.label, 1), -(s.end - s.start), s.start))
    kept: list[Redaction] = []
    for span in ranked:
        if span.end <= span.start:
            continue
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def apply(text: str, spans: Sequence[Redaction], replacement: Callable[[Redaction, str], str]) -> str:
    if not spans:
        return text
    out: list[str] = []
    pos = 0
    for span in spans:
        out.append(text[pos : span.start])
        out.append(replacement(span, text[span.start : span.end]))
        pos = span.end
    out.append(text[pos:])
    return "".join(out)


def marker(span: Redaction, _: str) -> str:
    return MARKER.format(label=span.label)


def _by_segment(spans: Iterable[Redaction]) -> dict[int, list[Redaction]]:
    out: dict[int, list[Redaction]] = {}
    for span in spans:
        out.setdefault(span.segment, []).append(span)
    return out


def redact_views(
    segments: Sequence[Segment],
    enforced: Sequence[Redaction],
    every: Sequence[Redaction],
    vault: GuardVault,
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """(upstream view with enforce redactions, scrubbed view with all redactions)

    placeholders are allocated in message order, left to right, so a conversation prefix numbers the same.
    """
    upstream_spans = {i: merge(s) for i, s in _by_segment(enforced).items()}
    scrubbed_spans = {i: merge(s) for i, s in _by_segment(every).items()}
    for seg in segments:
        for span in sorted(
            [*upstream_spans.get(seg.index, ()), *scrubbed_spans.get(seg.index, ())], key=lambda s: s.start
        ):
            vault.add(span.label, seg.text[span.start : span.end])

    def to_placeholder(span: Redaction, value: str) -> str:
        return vault.add(span.label, value)

    def view(spans: dict[int, list[Redaction]]) -> tuple[Segment, ...]:
        return tuple(
            replace(seg, text=apply(seg.text, spans[seg.index], to_placeholder), inspect=None, decoded=())
            if seg.index in spans
            else seg
            for seg in segments
        )

    return view(upstream_spans), view(scrubbed_spans)
