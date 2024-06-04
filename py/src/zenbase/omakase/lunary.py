from asyncer import asyncify
import lunary
from zenbase.asyncio import amap, pmap, syncify

from zenbase.functional import LMDemo, LMFunction
from zenbase.numerical import ExperimentResult, ExperimentRunner, maximize_score


class LunaryZen:
    @staticmethod
    def dataset_to_demos(dataset: list[lunary.DatasetItem]) -> list[LMDemo]:
        return [LMDemo(input=item.input, output=item.ideal_output) for item in dataset]

    def evaluate_numeric[
        Params: dict, Response: dict
    ](
        self,
        *args,
        checklist: str,
        dataset: list[lunary.DatasetItem],
        concurrency: int = 20,
        **kwargs,
    ) -> ExperimentRunner:
        def run_experiment(
            function: LMFunction[Params, Response]
        ) -> ExperimentResult[Params, Response]:
            responses = pmap(
                function.call_sync,
                [{"input": item.input} for item in dataset],
                concurrency=concurrency,
            )

            def evaluate(item: lunary.DatasetItem, response: Response):
                return lunary.evaluate(
                    checklist,
                    input={"input": item.input},
                    output=response,
                    ideal_output=item.ideal_output,
                    *args,
                    **kwargs,
                )

            eval_results = pmap(
                asyncify(evaluate),
                dataset,
                responses,
                concurrency=concurrency,
            )

            # TODO: Go from (passed, results) tuples to evals dict

            return ExperimentResult(function)

        return run_experiment
