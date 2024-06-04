from copy import copy
from functools import partial
from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable
from collections import deque
import inspect

from metadict import MetaDict
from pksuid import PKSUID


from zenbase.asyncio import asyncify, syncify


class LMDemo[Params: dict, Response: dict](BaseModel):
    params: Params
    response: Response


class LMZenbase[Params: dict, Response: dict](BaseModel):
    instructions: list[str] = Field(default_factory=list)
    dos: list[str] = Field(default_factory=list)
    donts: list[str] = Field(default_factory=list)
    demos: list[LMDemo[Params, Response]] = Field(default_factory=list)


class LMRequest[Params: dict, Response: dict](BaseModel):
    id: PKSUID = Field(default_factory=partial(PKSUID, "request"))
    zenbase: LMZenbase[Params, Response]
    params: Params = Field(default_factory=MetaDict)


class LMFunctionCall[Params: dict, Response: dict](BaseModel):
    function: "LMFunction[Params, Response]"
    request: LMRequest[Params, Response]
    response: Response
    id: PKSUID = Field(default_factory=partial(PKSUID, "call"))


type LMCallable[Params: dict, Response] = Callable[
    [LMRequest[Params, Response]],
    Response,
]


class LMFunction[Params: dict, Response: dict]:
    id: PKSUID
    sync_fn: LMCallable[Params, Response]
    async_fn: LMCallable[Params, Awaitable[Response]]
    __name__: str
    __qualname__: str
    __doc__: str
    __signature__: inspect.Signature
    zenbase: LMZenbase[Params, Response]
    history: deque[LMFunctionCall[Params, Response]]

    def __init__(
        self,
        fn: LMCallable[Params, Response] | LMCallable[Params, Awaitable[Response]],
        request_defaults: LMRequest[Params, Response] | None = None,
        maxhistory: int = 128,
    ):
        self.id = PKSUID("fn")
        self.sync_fn = syncify(fn)
        self.async_fn = asyncify(fn)

        self.__name__ = getattr(fn, "__name__", "zenbase_lm_fn")
        self.__qualname__ = getattr(fn, "__qualname__", "zenbase_lm_fn")
        self.__doc__ = getattr(fn, "__doc__", "")
        self.__signature__ = inspect.signature(fn)

        self.zenbase = request_defaults or LMRequest()
        self.history = deque([], maxlen=maxhistory)

    def refine(
        self, zenbase: LMZenbase[Params, Response] | None = None
    ) -> "LMFunction[Params, Response]":
        dup = copy(self)
        dup.id = PKSUID("fn")
        dup.zenbase = zenbase or self.zenbase.model_copy()
        dup.history = deque([], maxlen=self.history.maxlen)
        return dup

    def prepare_request(self, params: Params) -> LMRequest[Params, Response]:
        return LMRequest(zenbase=self.zenbase, params=params)

    def process_response(
        self, request: LMRequest[Params, Response], response: Response
    ) -> Response:
        self.history.append(
            LMFunctionCall(
                function=self,
                request=request,
                response=response,
            ),
        )
        return response

    async def __call__(
        self,
        params: Params,
    ) -> Response:
        request = self.prepare_request(params)
        response = await self.async_fn(request)
        return self.process_response(request, response)

    def call_sync(self, params: Params) -> Response:
        request = self.prepare_request(params)
        response = self.sync_fn(request)
        return self.process_response(request, response)


def deflm[
    Params: dict,
    Response: dict,
](
    function: (
        LMCallable[Params, Response] | LMCallable[Params, Awaitable[Response]] | None
    ) = None,
    request_defaults: LMRequest[Params, Response] | None = None,
    maxhistory: int = 128,
) -> LMFunction[Params, Response]:
    if function is None:
        return partial(deflm, request_defaults=request_defaults, maxhistory=maxhistory)

    if isinstance(function, LMFunction):
        return function.refine()

    return LMFunction(function, request_defaults, maxhistory)


LMFunctionCall.model_rebuild()
