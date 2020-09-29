import asyncio
from functools import partial
from typing import Any, Callable, List


async def run_in_background(
    func: Callable, *args: Any, **kwargs: Any
) -> Any:
    """
    Run a function in the background to prevent blocking the main thread.  Functions will be executed using the default
    thread pool executor of the event loop.  By default, event loops use a ``concurrent.futures.ThreadPoolExecutor``,
    however it is recommended to override use ``concurrent.futures.ProcessPoolExecutor`` instead:

    ``asyncio.get_event_loop().set_default_executor(concurrent.futures.ProcessPoolExecutor)``

    Ref: https://github.com/encode/starlette/blob/master/starlette/concurrency.py#L21-L34
    """
    loop = asyncio.get_event_loop()
    func = partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, func)


def chunks(lst: List, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]