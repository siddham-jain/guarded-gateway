"""recorded remote-detector answers, so the ci gate replays a live run without network or keys"""

from pathlib import Path

import httpx2
import orjson

from gg.core.jsonutil import loads, sha256_hex

MISSING_STATUS = 599


def _key(request: httpx2.Request) -> str:
    return sha256_hex(request.url.path.encode() + b"\n" + request.content)


class Cassette:
    """response bodies keyed by a hash of the request; prompts are never stored"""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, str] = loads(path.read_bytes()) if path.is_file() else {}
        self._seen: dict[str, str] = {}
        self._asked: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def replay(self) -> httpx2.AsyncBaseTransport:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = self._entries.get(_key(request))
            if body is None:
                # the items or the question changed since the recording; rerun with --live
                return httpx2.Response(MISSING_STATUS)
            return httpx2.Response(200, content=body.encode())

        return httpx2.MockTransport(handler)

    def record(self) -> httpx2.AsyncBaseTransport:
        return _Recorder(self._seen, self._asked)

    def save(self) -> None:
        """keeps only what this run asked; a call that failed this time keeps its earlier answer"""
        merged = {k: self._seen.get(k, self._entries.get(k)) for k in sorted(self._asked)}
        kept = {k: v for k, v in merged.items() if v is not None}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(orjson.dumps(kept, option=orjson.OPT_INDENT_2) + b"\n")


class _Recorder(httpx2.AsyncBaseTransport):
    def __init__(self, seen: dict[str, str], asked: set[str]) -> None:
        self._inner = httpx2.AsyncHTTPTransport()
        self._seen = seen
        self._asked = asked

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        self._asked.add(_key(request))
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        if response.status_code == 200:
            self._seen[_key(request)] = body.decode()
        headers = [(k, v) for k, v in response.headers.raw if k.lower() != b"content-encoding"]
        return httpx2.Response(response.status_code, headers=headers, content=body)

    async def aclose(self) -> None:
        await self._inner.aclose()
