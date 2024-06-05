from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random

from pyee.asyncio import AsyncIOEventEmitter

from zenbase.types import LMFunction
from zenbase.utils import alist, get_seed, syncify


@dataclass(kw_only=True)
class LMOptim[Inputs: dict, Outputs: dict](ABC):
    random: Random = field(default_factory=lambda: Random(get_seed()))
    events: AsyncIOEventEmitter = field(default_factory=AsyncIOEventEmitter)

    @abstractmethod
    def train(self, function: LMFunction[Inputs, Outputs], *args, **kwargs):
        return syncify(alist)(self.atrain(function, *args, **kwargs))

    @abstractmethod
    async def atrain(
        self, function: LMFunction[Inputs, Outputs], *args, **kwargs
    ) -> LMFunction[Inputs, Outputs]: ...
