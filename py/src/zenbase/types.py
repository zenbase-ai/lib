from collections import deque
from copy import copy
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Awaitable, Callable
import inspect


from zenbase.utils import asyncify, id_gen, syncify


@dataclass(frozen=True)
class LMDemo[Params: dict, Response: dict]:
    params: Params
    response: Response

    def __hash__(self):
        return hash((frozenset(self.params.items()), frozenset(self.response.items())))


@dataclass(frozen=True)
class LMZenbase[Params: dict, Response: dict]:
    instructions: list[str] = field(default_factory=list)
    dos: list[str] = field(default_factory=list)
    donts: list[str] = field(default_factory=list)
    demos: list[LMDemo[Params, Response]] = field(default_factory=list)


@dataclass(frozen=True)
class LMRequest[Params: dict, Response: dict]:
    zenbase: LMZenbase[Params, Response]
    params: Params = field(default_factory=dict)
    id: str = field(default_factory=id_gen("request"))


@dataclass(frozen=True)
class LMCall[Params: dict, Response: dict]:
    function: "LMFunction[Params, Response]"
    request: LMRequest[Params, Response]
    response: Response
    id: str = field(default_factory=id_gen("call"))


type LMCallable[Params: dict, Response] = Callable[
    [LMRequest[Params, Response]],
    Response,
]


class LMFunction[Params: dict, Response: dict]:
    gen_id = staticmethod(id_gen("fn"))

    id: str
    sync_fn: LMCallable[Params, Response]
    async_fn: LMCallable[Params, Awaitable[Response]]
    __name__: str
    __qualname__: str
    __doc__: str
    __signature__: inspect.Signature
    zenbase: LMZenbase
    history: deque[LMCall[Params, Response]]

    def __init__(
        self,
        fn: LMCallable[Params, Response] | LMCallable[Params, Awaitable[Response]],
        zenbase: LMZenbase | None = None,
        maxhistory: int = 128,
    ):
        self.id = self.gen_id()
        self.sync_fn = syncify(fn)
        self.async_fn = asyncify(fn)

        self.__name__ = getattr(fn, "__name__", "zenbase_lm_fn")
        self.__qualname__ = getattr(fn, "__qualname__", "zenbase_lm_fn")
        self.__doc__ = getattr(fn, "__doc__", "")
        self.__signature__ = inspect.signature(fn)

        self.zenbase = zenbase or LMZenbase()
        self.history = deque([], maxlen=maxhistory)

    def refine(
        self, zenbase: LMZenbase | None = None
    ) -> "LMFunction[Params, Response]":
        dup = copy(self)
        dup.id = self.gen_id()
        dup.zenbase = zenbase or replace(self.zenbase)
        dup.history = deque([], maxlen=self.history.maxlen)
        return dup

    def prepare_request(self, params: Params) -> LMRequest[Params, Response]:
        return LMRequest(zenbase=self.zenbase, params=params)

    def process_response(
        self, request: LMRequest[Params, Response], response: Response
    ) -> Response:
        self.history.append(
            LMCall(
                function=self,
                request=request,
                response=response,
            ),
        )
        return response

    async def __call__(
        self,
        params: Params = {},
    ) -> Response:
        request = self.prepare_request(params)
        response = await self.async_fn(request)
        return self.process_response(request, response)

    def call_sync(self, params: Params = {}) -> Response:
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
