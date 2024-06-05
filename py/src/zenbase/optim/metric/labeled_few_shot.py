from dataclasses import dataclass, field
from typing import cast

from zenbase.optim.abc import LMOptim
from zenbase.optim.metric.types import CandidateMetricEvaluator, CandidateMetricResult
from zenbase.types import LMDemo, LMFunction, LMZenbase
from zenbase.utils import amap, asyncify, syncify, tracer, get_logger


log = get_logger(__name__)


@dataclass(kw_only=True)
class LabeledFewShot[Params: dict, Response: dict](LMOptim):
    demoset: list[LMDemo[Params, Response]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert 1 <= self.shots <= len(self.demoset)

    @tracer.start_as_current_span("train")
    async def atrain(
        self,
        function: LMFunction[Params, Response],
        evaluator: CandidateMetricEvaluator[Params, Response],
        batch_size: int = 0,
        epochs: int = 1,
        concurrency: int = 1,
    ) -> LMFunction[Params, Response]:
        batch_size = batch_size or len(self.demoset)
        evaluate = asyncify(evaluator)

        score = float("-inf")
        best = function

        @tracer.start_as_current_span("run_experiment")
        async def run_candidate_zenbase(zenbase: LMZenbase):
            nonlocal score, best

            candidate_fn = function.refine(zenbase)
            result = cast(CandidateMetricResult, await evaluate(candidate_fn))

            self.events.emit("experiment", result)

            if result.evals["score"] > score:
                score = result.evals["score"]
                best = candidate_fn

            return result

        for _ in range(epochs):
            await amap(
                run_candidate_zenbase,
                self.candidates(best, batch_size),
                concurrency=concurrency,
            )

        return best

    def train(
        self,
        lmfn: LMFunction[Params, Response],
        evaluator: CandidateMetricEvaluator[Params, Response],
        batch_size: int = 0,
        epochs: int = 1,
        concurrency: int = 1,
    ) -> LMFunction[Params, Response]:
        return syncify(self.atrain)(lmfn, evaluator, batch_size, epochs, concurrency)

    def candidates(self, _: LMFunction[Params, Response], batch_size: int):
        if batch_size > len(self.demoset):
            log.warn(
                "Batch size is greater than the demos, using demoset size",
                demoset_size=len(self.demoset),
                batch_size=batch_size,
            )
            batch_size = len(self.demoset)

        for _ in range(batch_size):
            demos = tuple(self.random.sample(self.demoset, k=self.shots))
            yield LMZenbase(demos=demos)
