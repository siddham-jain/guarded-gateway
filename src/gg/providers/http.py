from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2

from gg.core.deployment import Timeouts
from gg.core.errors import ProviderError
from gg.providers.errors import transport_error

MAX_ERROR_BODY_BYTES = 65_536

type Classifier = Callable[[int, bytes, Mapping[str, str]], ProviderError]


@dataclass(frozen=True, slots=True)
class UpstreamRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes


def request_timeout(timeouts: Timeouts) -> httpx2.Timeout:
    # read is the inter-chunk stall timeout; ttft and the total deadline are enforced by the executor
    return httpx2.Timeout(connect=timeouts.connect_s, read=timeouts.inter_chunk_s, write=10.0, pool=2.0)


class HttpClientFactory:
    """one pooled client per (provider, base url); tests inject transports keyed by provider name"""

    def __init__(
        self,
        *,
        transports: Mapping[str, httpx2.AsyncBaseTransport] | None = None,
        default_transport: httpx2.AsyncBaseTransport | None = None,
        limits: httpx2.Limits | None = None,
    ) -> None:
        self._transports = dict(transports or {})
        self._default_transport = default_transport
        self._limits = limits or httpx2.Limits(
            max_connections=200, max_keepalive_connections=50, keepalive_expiry=30
        )
        self._clients: dict[tuple[str, str], httpx2.AsyncClient] = {}

    def client(self, provider: str, base_url: str) -> httpx2.AsyncClient:
        key = (provider, base_url)
        client = self._clients.get(key)
        if client is None:
            transport = self._transports.get(provider, self._default_transport)
            client = httpx2.AsyncClient(
                base_url=base_url,
                limits=self._limits,
                timeout=httpx2.Timeout(connect=3.0, read=60.0, write=10.0, pool=2.0),
                transport=transport,
                follow_redirects=False,
            )
            self._clients[key] = client
        return client

    async def aclose(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            await client.aclose()


async def _read_capped(response: httpx2.Response) -> bytes:
    parts: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        parts.append(chunk)
        size += len(chunk)
        if size >= MAX_ERROR_BODY_BYTES:
            break
    return b"".join(parts)[:MAX_ERROR_BODY_BYTES]


@asynccontextmanager
async def open_stream(
    client: httpx2.AsyncClient,
    up: UpstreamRequest,
    *,
    timeouts: httpx2.Timeout,
    classify: Classifier,
    provider: str,
    transport_kind: str = "retryable",
) -> AsyncGenerator[httpx2.Response]:
    """opens a streaming request; non-2xx raises the classified error, transport failures become status 0"""
    try:
        async with client.stream(
            up.method, up.url, headers=up.headers, content=up.body, timeout=timeouts
        ) as resp:
            if resp.status_code >= 400:
                raise classify(resp.status_code, await _read_capped(resp), resp.headers)
            yield resp
    except httpx2.TransportError as exc:
        kind = "fallback" if transport_kind == "fallback" else "retryable"
        raise transport_error(exc, provider=provider, kind=kind) from exc
