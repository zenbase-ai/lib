from typing import TYPE_CHECKING, Iterator

from zenbase.types import LMDemo, LMFunction
from zenbase.optim.metric.types import (
    MetricEvals,
    MetricExperimentResult,
    MetricExperimentEvaluator,
)
from zenbase.utils import random_name_gen

if TYPE_CHECKING:
    from langsmith import schemas


class ZenLangSmith:
    @staticmethod
    def examples_to_demos(examples: Iterator["schemas.Example"]) -> list[LMDemo]:
        return [LMDemo(params=e.inputs, response=e.outputs) for e in examples]

    @classmethod
    def metric_evaluator[
        Params: dict, Response: dict
    ](cls, **evaluate_kwargs) -> MetricExperimentEvaluator:
        from langsmith import evaluate

        metadata = evaluate_kwargs.pop("metadata", {})
        gen_random_name = random_name_gen(
            evaluate_kwargs.pop("experiment_prefix", None)
        )

        def run_experiment(
            function: LMFunction[Params, Response],
        ) -> MetricExperimentResult[Params, Response]:
            experiment_results = evaluate(
                function.call_sync,
                experiment_prefix=gen_random_name(),
                metadata={
                    **metadata,
                    **dict(function.zenbase),
                },
                **evaluate_kwargs,
            )

            if summary_results := experiment_results._summary_results["results"]:
                evals = cls._eval_results_to_evals(summary_results)
            else:
                evals = cls._experiment_results_to_evals(experiment_results)

            return MetricExperimentResult(function, evals)

        return run_experiment

    @staticmethod
    def _experiment_results_to_evals(experiment_results: list) -> MetricEvals:
        total = sum(
            res["evaluation_results"]["results"][0].score
            for res in experiment_results._results
        )
        count = len(experiment_results._results)
        mean = total / count
        return {"score": mean}

    @staticmethod
    def _eval_results_to_evals(eval_results: list) -> MetricEvals:
        if not eval_results:
            raise ValueError("No evaluation results")

        return {
            "score": eval_results[0].score,
            **{r.key: r.dict() for r in eval_results},
        }
