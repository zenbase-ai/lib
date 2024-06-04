from os import getenv
from random import Random
from typing import Generator

from zenbase.functional import LMDemo, LMZenbase


class LabeledFewShot:
    @staticmethod
    def candidates[
        Params: dict, Response: dict
    ](
        demos: list[LMDemo[Params, Response]],
        shots: int = 5,
        samples: int = 100,
        seed: int | None = None,
    ) -> Generator[LMZenbase[Params, Response], None, None]:
        assert len(demos) >= shots, f"Need at least {shots} demos"

        if seed is None:
            seed = int(getenv("RANDOM_SEED", 42))

        rng = Random(seed)
        seen = set()

        for _ in range(samples):
            demos = rng.sample(demos, k=shots)
            if demos not in seen:
                seen.add(demos)
                yield LMZenbase(demos=demos)
