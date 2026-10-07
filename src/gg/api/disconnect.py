import asyncio
from collections.abc import Awaitable

from starlette.types import Receive


class ClientDisconnectedError(Exception):
    pass


async def wait_for_disconnect(receive: Receive) -> None:
    # only valid once the request body has been consumed
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


async def run_with_disconnect_watch[T](receive: Receive, work: Awaitable[T]) -> T:
    """races work against a client disconnect; on disconnect work is cancelled and awaited"""
    task = asyncio.ensure_future(work)
    watcher = asyncio.create_task(wait_for_disconnect(receive))
    try:
        await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)
    if task.cancelled():
        raise ClientDisconnectedError
    return task.result()
