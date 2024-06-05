import lunary

from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.types import LMDemo, LMFunction
from zenbase.utils import pmap


class ZenLunary:
    @staticmethod
    def dataset_to_demos(dataset: list[lunary.DatasetItem]) -> list[LMDemo]:
        return [LMDemo(input=item.input, output=item.ideal_output) for item in dataset]

    @classmethod
    def metric_evaluator[
        Params: dict, Response: dict
    ](
        cls,
        *args,
        checklist: str,
        evalset: list[lunary.DatasetItem],
        concurrency: int = 20,
        **kwargs,
    ) -> CandidateMetricEvaluator:
        def evaluate_metric(
            function: LMFunction[Params, Response]
        ) -> CandidateMetricResult[Params, Response]:
            def run_and_evaluate(item: lunary.DatasetItem):
                response = function.call_sync({"input": item.input})
                passed, results = lunary.evaluate(
                    checklist,
                    input={"input": item.input},
                    output=response,
                    ideal_output=item.ideal_output,
                    *args,
                    **kwargs,
                )
                return MetricEvals(score=int(passed), **results)

            evals = pmap(
                run_and_evaluate,
                evalset,
                concurrency=concurrency,
            )

            return CandidateMetricResult(function, evals)

        return evaluate_metric
