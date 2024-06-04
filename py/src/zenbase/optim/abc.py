from abc import ABC
from typing import AsyncGenerator

from zenbase.types import LMZenbase, LMFunction


class LMOptim[Params: dict, Response: dict](ABC):
    def reset(self): ...

    async def acandidates(
        self, function: LMFunction[Params, Response]
    ) -> AsyncGenerator[LMZenbase, None]:
        raise NotImplementedError()
