from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from gg.providers.sse import SSEEvent, SSEParser, aiter_sse


def parse(*chunks: bytes) -> list[SSEEvent]:
    parser = SSEParser()
    events: list[SSEEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    return events + parser.flush()


def test_single_event() -> None:
    assert parse(b"data: hello\n\n") == [SSEEvent("hello")]


def test_line_endings_crlf_and_cr() -> None:
    assert parse(b"data: a\r\n\r\ndata: b\r\rdata: c\n\n") == [SSEEvent("a"), SSEEvent("b"), SSEEvent("c")]


def test_crlf_split_across_reads_is_one_line_end() -> None:
    assert parse(b"data: a\r", b"\n\r\n") == [SSEEvent("a")]


def test_multiline_data_is_joined_with_newline() -> None:
    assert parse(b"data: one\ndata: two\n\n") == [SSEEvent("one\ntwo")]


def test_comments_are_ignored_and_empty_events_dropped() -> None:
    assert parse(b": keep-alive\n\n: other\ndata: x\n\n\n\n") == [SSEEvent("x")]


def test_only_one_leading_space_is_stripped() -> None:
    assert parse(b"data:  two spaces\ndata:none\n\n") == [SSEEvent(" two spaces\nnone")]


def test_event_and_id_fields() -> None:
    events = parse(b"event: message_start\nid: 7\ndata: {}\n\ndata: next\n\n")
    assert events == [SSEEvent("{}", "message_start", "7"), SSEEvent("next", "message", "7")]


def test_split_multibyte_utf8() -> None:
    raw = "data: héllo 🙂\n\n".encode()
    cut = raw.index(b"\xf0") + 2
    assert parse(raw[:cut], raw[cut:]) == [SSEEvent("héllo 🙂")]


def test_trailing_event_without_blank_line_flushed_at_eof() -> None:
    assert parse(b"data: [DONE]") == [SSEEvent("[DONE]")]


def test_leading_bom_is_skipped() -> None:
    assert parse("﻿data: x\n\n".encode()) == [SSEEvent("x")]


def test_field_without_colon_and_unknown_fields() -> None:
    assert parse(b"data\nretry: 10\nfoo: bar\n\n") == [SSEEvent("")]


async def test_aiter_sse_over_async_chunks() -> None:
    async def chunks():
        yield b"data: a\n"
        yield b"\ndata: b"

    assert [e async for e in aiter_sse(chunks())] == [SSEEvent("a"), SSEEvent("b")]


STREAM = (
    b': comment\r\ndata: {"a":1}\r\n\r\nevent: x\ndata: line1\ndata: line2\n\n'
    + "data: ünïcödé ✓\r\r".encode()
    + b"data: [DONE]\n\n"
)


@given(st.lists(st.integers(min_value=0, max_value=len(STREAM)), max_size=12))
def test_any_partition_yields_identical_events(cuts: list[int]) -> None:
    points = sorted({0, len(STREAM), *cuts})
    pieces = [STREAM[a:b] for a, b in pairwise(points)]
    assert parse(*pieces) == parse(STREAM)
