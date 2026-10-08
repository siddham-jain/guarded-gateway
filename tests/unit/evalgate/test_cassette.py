from pathlib import Path

import httpx2

from gg.evalgate.cassette import MISSING_STATUS, Cassette, _key  # pyright: ignore[reportPrivateUsage]


def request(body: str) -> httpx2.Request:
    return httpx2.Request("POST", "https://jev.test/v1/systemone", content=body.encode())


async def test_replay_answers_recorded_requests_and_refuses_unknown_ones(tmp_path: Path) -> None:
    path = tmp_path / "jev.json"
    path.write_text(f'{{"{_key(request("known"))}": "{{\\"ok\\": true}}"}}')
    cassette = Cassette(path)
    assert len(cassette) == 1
    async with httpx2.AsyncClient(transport=cassette.replay()) as http:
        known = await http.send(request("known"))
        unknown = await http.send(request("changed prompt"))
    assert (known.status_code, known.json()) == (200, {"ok": True})
    assert unknown.status_code == MISSING_STATUS


def test_save_keeps_what_was_asked_and_earlier_answers_for_failed_calls(tmp_path: Path) -> None:
    path = tmp_path / "cassettes" / "jev.json"
    path.parent.mkdir()
    old, stale, fresh = _key(request("old")), _key(request("stale")), _key(request("fresh"))
    path.write_text(f'{{"{old}": "old answer", "{stale}": "never asked again"}}')
    cassette = Cassette(path)
    # this run asked "old" (the call failed, nothing seen) and "fresh" (answered)
    cassette._asked.update({old, fresh})  # pyright: ignore[reportPrivateUsage]
    cassette._seen[fresh] = "fresh answer"  # pyright: ignore[reportPrivateUsage]
    cassette.save()
    assert Cassette(path)._entries == {old: "old answer", fresh: "fresh answer"}  # pyright: ignore[reportPrivateUsage]
