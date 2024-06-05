from dataclasses import dataclass, field
from typing import cast

from zenbase.optim.abc import LMOptim
from zenbase.optim.metric.types import MetricExperimentEvaluator, MetricExperimentResult
from zenbase.types import LMDemo, LMFunction, LMZenbase
from zenbase.utils import alist, amap, asyncify, syncify, tracer, logger


log = logger.getChild(__name__)


@dataclass(kw_only=True)
class LabeledFewShot[Params: dict, Response: dict](LMOptim):
    demoset: list[LMDemo[Params, Response]]
    shots: int = field(default=5)

    def __post_init__(self):
        assert self.shots <= len(self.demoset), "Insufficient samples demos"

    @tracer.start_as_current_span("candidates")
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

    async def acandidates(self, _: LMFunction[Params, Response], batch_size: int):
        for candidate in self.candidates(_, batch_size):
            yield candidate

    def compile(
        self,
        lmfn: LMFunction[Params, Response],
        evaluator: MetricExperimentEvaluator[Params, Response],
        batch_size: int = 0,
        epochs: int = 1,
    ):
        return syncify(self.acompile)(lmfn, evaluator, batch_size, epochs)

    @tracer.start_as_current_span("compile")
    async def acompile(
        self,
        function: LMFunction[Params, Response],
        evaluator: MetricExperimentEvaluator[Params, Response],
        batch_size: int = 0,
        epochs: int = 1,
    ) -> LMFunction[Params, Response]:
        run_experiment = asyncify(evaluator)

        score = float("-inf")
        best = function

        @tracer.start_as_current_span("run_experiment")
        async def do_experiment(zenbase: LMZenbase):
            nonlocal score, best

            candidate_fn = function.refine(zenbase)
            result = cast(MetricExperimentResult, await run_experiment(candidate_fn))

            if result.evals["score"] > score:
                score = result.evals["score"]
                best = candidate_fn

            return result

        batch_size = batch_size or len(self.demoset)
        for _ in range(epochs):
            candidates = list(self.candidates(best, batch_size))
            await amap(do_experiment, candidates, concurrency=1)

        return best
