import pickle
from typing import Any

import pytest

from gg.core.schema import ChatRequest
from gg.core.vault import PlaceholderVault
from gg.guardrails.base import Redaction, Segment
from gg.guardrails.redact import MARKER, apply, marker, merge, redact_views
from gg.guardrails.segments import PLACEHOLDER_HINT, extract_segments, with_placeholder_hint, write_back
from gg.guardrails.vault import PLACEHOLDER_RE, GuardVault


def test_vault_numbers_per_label_and_reuses_canonical_values() -> None:
    vault = GuardVault()
    a = vault.add("EMAIL", "Bob@X.com")
    assert a == "[EMAIL_1]"
    assert vault.add("EMAIL", "bob@x.com") == a
    assert vault.add("EMAIL", "eve@x.com") == "[EMAIL_2]"
    assert vault.add("PHONE", "+44 20 1234 5678") == "[PHONE_1]"
    assert vault.add("PHONE", "+44-20-1234-5678") == "[PHONE_1]"
    assert vault.resolve(a) == "Bob@X.com"
    assert vault.is_vault_value("EMAIL", "BOB@x.com")
    assert not vault.is_vault_value("EMAIL", "mallory@x.com")
    assert vault.summary() == {"EMAIL": 2, "PHONE": 1}


def test_vault_never_issues_a_placeholder_the_client_typed() -> None:
    vault = GuardVault()
    vault.reserve(["literally [EMAIL_1] and [email_2]"])
    assert vault.add("EMAIL", "a@x.com") == "[EMAIL_3]"
    # a typed literal is never restored, even case-insensitively
    assert vault.restore("[EMAIL_1] [email_2] [EMAIL_3]") == "[EMAIL_1] [email_2] a@x.com"


def test_restore_exact_then_case_insensitive_and_json_escaped() -> None:
    vault = GuardVault()
    vault.add("EMAIL", 'quote"me@x.com')
    assert vault.restore("hi [EMAIL_1]") == 'hi quote"me@x.com'
    assert vault.restore("hi [Email_1]") == 'hi quote"me@x.com'
    assert vault.restore('{"to": "[EMAIL_1]"}', json_escape=True) == '{"to": "quote\\"me@x.com"}'
    assert vault.stats.exact == 2
    assert vault.stats.case_insensitive == 1


def test_unknown_placeholders_and_markers_stay() -> None:
    vault = GuardVault()
    vault.add("EMAIL", "a@x.com")
    assert vault.restore("[EMAIL_7] [REDACTED:EMAIL]") == "[EMAIL_7] [REDACTED:EMAIL]"
    assert vault.stats.unmatched == 1
    assert PLACEHOLDER_RE.search(MARKER.format(label="EMAIL")) is None


@pytest.mark.parametrize(
    ("text", "held"),
    [("hello", 0), ("hello [", 1), ("hi [EMA", 4), ("x [EMAIL_", 7), ("x [EMAIL_1", 8), ("x [EMAIL_1]", 0)],
)
def test_partial_suffix_len(text: str, held: int) -> None:
    assert GuardVault.partial_suffix_len(text) == held


def test_vault_is_never_serialised_or_printed() -> None:
    vault = GuardVault()
    vault.add("EMAIL", "secret@x.com")
    with pytest.raises(TypeError):
        pickle.dumps(vault)
    assert "secret@x.com" not in repr(vault)


def test_adopt_keeps_core_vault_entries() -> None:
    core = PlaceholderVault()
    core.add("EMAIL", "a@x.com")
    adopted = GuardVault.adopt(core)
    assert adopted.placeholders() == core.placeholders()
    assert GuardVault.adopt(adopted) is adopted


def _request(**extra: Any) -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys secret"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "u"}},
                        {"type": "text", "text": "b"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "c",
                    "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": '{"x": 1}'}}],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "d"},
            ],
            **extra,
        }
    )


def test_extract_skips_system_and_maps_every_text_location() -> None:
    segs = extract_segments(_request())
    assert [(s.role, s.kind, s.msg, s.part, s.tool_call, s.text) for s in segs] == [
        ("user", "content", 1, 0, -1, "a"),
        ("user", "content", 1, 2, -1, "b"),
        ("assistant", "content", 2, 0, -1, "c"),
        ("assistant", "tool_args", 2, 0, 0, '{"x": 1}'),
        ("tool", "tool_result", 3, 0, -1, "d"),
    ]


def test_write_back_rewrites_only_changed_segments_and_keeps_unknown_fields() -> None:
    request = _request(custom_field={"keep": True})
    segs = extract_segments(request)
    assert write_back(request, segs, segs) is request
    changed = tuple(s if s.index != 3 else Segment(**{**_fields(s), "text": '{"x": 2}'}) for s in segs)
    changed = tuple(s if s.index != 1 else Segment(**{**_fields(s), "text": "B"}) for s in changed)
    out = write_back(request, segs, changed)
    assert out.messages[1].text() == "aB"
    assert out.messages[2].tool_calls is not None
    assert out.messages[2].tool_calls[0].function.arguments == '{"x": 2}'
    assert out.messages[0] == request.messages[0]
    assert (out.model_extra or {})["custom_field"] == {"keep": True}


def _fields(seg: Segment) -> dict[str, Any]:
    return {f: getattr(seg, f) for f in Segment.__dataclass_fields__}


def test_placeholder_hint_appends_to_system_or_inserts_one() -> None:
    with_system = with_placeholder_hint(_request())
    assert with_system.messages[0].text().endswith(PLACEHOLDER_HINT)
    bare = ChatRequest.model_validate({"model": "m", "messages": [{"role": "user", "content": "x"}]})
    hinted = with_placeholder_hint(bare)
    assert hinted.messages[0].role == "system"
    assert hinted.messages[0].text() == PLACEHOLDER_HINT


def test_merge_prefers_secret_then_longer_span() -> None:
    spans = [
        Redaction(0, 10, 20, "EMAIL"),
        Redaction(0, 0, 30, "SECRET"),
        Redaction(0, 40, 45, "PHONE"),
        Redaction(0, 38, 50, "PHONE"),
    ]
    assert [(s.start, s.end, s.label) for s in merge(spans)] == [(0, 30, "SECRET"), (38, 50, "PHONE")]


def test_apply_replaces_left_to_right() -> None:
    text = "mail a@x.com or b@x.com"
    spans = merge([Redaction(0, 16, 23, "EMAIL"), Redaction(0, 5, 12, "EMAIL")])
    assert apply(text, spans, marker) == "mail [REDACTED:EMAIL] or [REDACTED:EMAIL]"


def test_redact_views_split_enforced_and_shadow_with_stable_numbering() -> None:
    segs = (
        Segment(index=0, role="user", kind="content", msg=0, text="a@x.com and 555-123-4567"),
        Segment(index=1, role="user", kind="content", msg=1, text="again a@x.com"),
    )
    email = [Redaction(0, 0, 7, "EMAIL"), Redaction(1, 6, 13, "EMAIL")]
    phone = [Redaction(0, 12, 24, "PHONE")]
    vault = GuardVault()
    upstream, scrubbed = redact_views(segs, email, [*email, *phone], vault)
    assert [s.text for s in upstream] == ["[EMAIL_1] and 555-123-4567", "again [EMAIL_1]"]
    assert [s.text for s in scrubbed] == ["[EMAIL_1] and [PHONE_1]", "again [EMAIL_1]"]
