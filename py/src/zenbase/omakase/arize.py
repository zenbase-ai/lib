from typing import Callable, Iterable

from phoenix.evals import LLMEvaluator, run_evals
import pandas as pd
import phoenix as px
from zenbase.numerical import (
    ExperimentResult,
    maximize_score,
)

from zenbase.functional import (
    LMCallable,
    LMDemo,
    LMFunction,
    LMZenbase,
    deflm,
)
from zenbase.asyncio import amap


type ArizeEvaluator = Callable[[px.Client], tuple[float, pd.DataFrame]]


def json_denormalize(df: pd.DataFrame, prefix: str, depth=1):
    assert depth == 1, "Only 1 level of denormalization is supported"

    columns = [c for c in df.columns if c.startswith(prefix)]
    return {
        c[len(prefix) :]: r[c]
        for c in columns
        for r in df[columns].to_dict(orient="records")
    }


class PhoenixZen:
    @staticmethod
    def df_to_demos[
        Params: dict, Response: dict
    ](df: pd.DataFrame) -> list[LMDemo[Params, Response]]:
        raise NotImplementedError()

    @classmethod
    def numeric_evals[
        Params: dict,
        Response: dict,
    ](
        cls,
        dataset: pd.DataFrame,
        evaluators: list[LLMEvaluator],
        concurrency: int = 20,
        *args,
        **kwargs,
    ):
        async def run_experiment(
            function: LMFunction[Params, Response],
        ) -> ExperimentResult[Params, Response]:
            nonlocal dataset
            candidate_df = dataset.copy()
            responses = await amap(
                function,
                candidate_df["attributes.input.value"].to_list(),
                concurrency=concurrency,
            )
            candidate_df["attributes.output.value"] = responses

            dfs = run_evals(
                candidate_df, evaluators, *args, concurrency=concurrency, **kwargs
            )
            evals = {"score": dfs[0].score.mean()}
            evals = {
                evaluator.__name__: df.score.mean()
                for evaluator, df in zip(evaluators, dfs)
            }

            return ExperimentResult(function, evals)

        return run_experiment

    @classmethod
    def maximize_score[
        Params: dict,
        Response: dict,
    ](
        cls,
        fn: LMCallable[Params, Response] | LMFunction[Params, Response],
        candidates: Iterable[LMZenbase[Params, Response]],
        dataset: pd.DataFrame,
        evaluators: list[LLMEvaluator],
        *args,
        concurrency: int = 20,
        **kwargs,
    ):
        return maximize_score(
            deflm(fn),
            candidates,
            cls.numeric_evals(dataset, evaluators, concurrency, *args, **kwargs),
        )
