import asyncio

import pytest
from starlette.types import Message, Receive

from gg.api.disconnect import ClientDisconnectedError, run_with_disconnect_watch, wait_for_disconnect


def _receiver(disconnect: asyncio.Event) -> Receive:
    async def receive() -> Message:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    return receive


async def test_returns_result_and_stops_watcher() -> None:
    async def work() -> int:
        return 7

    assert await run_with_disconnect_watch(_receiver(asyncio.Event()), work()) == 7


async def test_propagates_errors() -> None:
    async def work() -> int:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        await run_with_disconnect_watch(_receiver(asyncio.Event()), work())


async def test_disconnect_cancels_work() -> None:
    disconnect = asyncio.Event()
    cancelled = asyncio.Event()

    async def work() -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 1

    asyncio.get_running_loop().call_later(0.02, disconnect.set)
    with pytest.raises(ClientDisconnectedError):
        await run_with_disconnect_watch(_receiver(disconnect), work())
    assert cancelled.is_set()


async def test_wait_for_disconnect_skips_other_messages() -> None:
    messages: list[Message] = [{"type": "http.request", "body": b""}, {"type": "http.disconnect"}]

    async def receive() -> Message:
        return messages.pop(0)

    await wait_for_disconnect(receive)
    assert messages == []
