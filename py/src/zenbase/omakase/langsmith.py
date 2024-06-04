from typing import Iterable, Iterator

from langsmith import evaluate, schemas
from langsmith.evaluation._runner import _get_random_name

from zenbase.functional import (
    LMCallable,
    LMDemo,
    LMFunction,
    LMZenbase,
    deflm,
)
from zenbase.numerical import Evals, ExperimentResult, ExperimentRunner, maximize_score


class LangSmithZen:
    @staticmethod
    def examples_to_demos(examples: Iterator[schemas.Example]) -> list[LMDemo]:
        return [LMDemo(params=e.inputs, response=e.outputs) for e in examples]

    @classmethod
    def evaluate_numeric[
        Params: dict, Response: dict
    ](cls, **evaluate_kwargs) -> ExperimentRunner:
        metadata = evaluate_kwargs.pop("metadata", {})

        experiment_prefix = "zenbase"
        if prefix := evaluate_kwargs.pop("experiment_prefix", None):
            experiment_prefix += f"-{prefix}"

        def run_experiment(
            function: LMFunction[Params, Response],
        ) -> ExperimentResult[Params, Response]:
            experiment_results = evaluate(
                function.call_sync,
                experiment_prefix=f"{experiment_prefix}-{_get_random_name()}",
                metadata={
                    **metadata,
                    **function.zenbase.model_dump(),
                },
                **evaluate_kwargs,
            )

            if summary_results := experiment_results._summary_results["results"]:
                evals = cls._eval_results_to_evals(summary_results)
            else:
                evals = cls._experiment_results_to_evals(experiment_results)

            return ExperimentResult(function, evals)

        return run_experiment

    @classmethod
    def maximize_score[
        Params: dict, Response: dict
    ](
        cls,
        fn: LMCallable[Params, Response] | LMFunction[Params, Response],
        candidates: Iterable[LMZenbase[Params, Response]],
        **evaluate_kwargs,
    ):
        return maximize_score(
            deflm(fn),
            candidates,
            cls.evaluate_numeric(**evaluate_kwargs),
        )

    @staticmethod
    def _experiment_results_to_evals(experiment_results: list) -> Evals:
        total = sum(
            res["evaluation_results"]["results"][0].score
            for res in experiment_results._results
        )
        count = len(experiment_results._results)
        mean = total / count
        return {"score": mean}

    @staticmethod
    def _eval_results_to_evals(eval_results: list) -> Evals:
        if not eval_results:
            raise ValueError("No evaluation results")

        return {
            "score": eval_results[0].score,
            **{r.key: r.dict() for r in eval_results},
        }
