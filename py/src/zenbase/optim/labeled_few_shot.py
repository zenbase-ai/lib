from dataclasses import dataclass, field
from math import factorial
from random import Random

from zenbase.optim.abc import LMOptim
from zenbase.types import LMDemo, LMFunction, LMZenbase
from zenbase.utils import alist, get_seed, syncify


@dataclass
class LabeledFewShot[Params: dict, Response: dict](LMOptim):
    demoset: list[LMDemo[Params, Response]]
    shots: int = field(default=5)
    samples: int = field(default=100)
    random: Random = field(default_factory=lambda: Random(get_seed()))

    _seen: set[tuple[LMDemo]] = field(default_factory=set)

    def __post_init__(self):
        assert self.shots <= len(self.demoset), "Insufficient samples demos"
        assert self.samples <= factorial(
            len(self.demoset)
        ), "Insufficient demos samples"

    def candidates(self, _function: LMFunction[Params, Response]):
        return syncify(alist)(self.acandidates(_function))

    async def acandidates(self, _function: LMFunction[Params, Response]):
        # TODO: Ensure self.samples is at least the number of unique demos
        for _ in range(self.samples):
            demos = tuple(self.random.sample(self.demoset, k=self.shots))
            if demos not in self._seen:
                yield LMZenbase(demos=demos)

                self._seen.add(demos)

    def reset(self):
        self._seen.clear()
