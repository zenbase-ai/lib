import asyncio
import functools
import inspect
from typing import Awaitable, Callable, ParamSpec, TypeVar

import asyncer

I_ParamSpec = ParamSpec("I_ParamSpec")
O_Retval = TypeVar("O_Retval")


async def asyncify(
    func: Callable[I_ParamSpec, O_Retval],
) -> Callable[I_ParamSpec, O_Retval]:
    if inspect.iscoroutinefunction(func):
        return func
    return asyncer.asyncify(func)


async def syncify(
    func: Callable[I_ParamSpec, O_Retval],
) -> Callable[I_ParamSpec, O_Retval]:
    if not inspect.iscoroutinefunction(func):
        return func
    return asyncer.syncify(func, raise_sync_error=False)


async def amap[
    O
](
    func: Callable[..., Awaitable[O]], iterable, *iterables, concurrency=float("inf")
) -> list[O]:
    assert concurrency >= 1, "Concurrency must be greater than 0"

    if concurrency == 1:
        return [await func(*args) for args in zip(iterable, *iterables)]

    if concurrency == float("inf"):
        return await asyncio.gather(
            *[func(*args) for args in zip(iterable, *iterables)]
        )

    semaphore = asyncio.Semaphore(concurrency)

    @functools.wraps(func)
    async def mapper(*args):
        async with semaphore:
            return await func(*args)

    return await asyncio.gather(*[mapper(*args) for args in zip(iterable, *iterables)])


def pmap[
    O
](func: Callable[..., O], iterable, *iterables, concurrency=float("inf")) -> list[O]:
    return syncify(amap)(asyncify(func), iterable, *iterables, concurrency)
