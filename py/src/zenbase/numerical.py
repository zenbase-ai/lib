from dataclasses import dataclass, field
from typing import Callable, Iterable, NamedTuple, TypedDict, cast

from zenbase.asyncio import amap, asyncify, syncify
from zenbase.functional import LMFunction, LMFunctionCall, LMZenbase


class Evals(TypedDict):
    score: float


class ExperimentRun(NamedTuple):
    call: LMFunctionCall
    evals: Evals


@dataclass
class ExperimentResult[Params: dict, Response: dict]:
    function: LMFunction[Params, Response]
    evals: Evals = field(default_factory=dict)


type ExperimentRunner[Params: dict, Response: dict] = Callable[
    [LMFunction[Params, Response]],
    ExperimentResult[Params, Response],
]


@dataclass
class NumericOptimizationResult[Params: dict, Response: dict]:
    function: LMFunction[Params, Response]
    experiments: list[ExperimentResult[Params, Response]]


type ExperimentEvaluator = Callable[[list[ExperimentRun]], Evals]


async def amaximize_score[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    candidates: Iterable[LMZenbase[Params, Response]],
    experimenter: ExperimentRunner[Params, Response],
    concurrency: int = 1,
) -> NumericOptimizationResult[Params, Response]:
    run_experiment = asyncify(experimenter)

    best_score = float("-inf")
    best_fn = None

    async def do_evaluate(zenbase: LMZenbase[Params, Response]):
        nonlocal best_score, best_fn

        candidate_fn = function.refine(zenbase)
        dataset_eval = cast(ExperimentResult, await run_experiment(candidate_fn))

        if dataset_eval.evals["score"] > best_score:
            best_score = dataset_eval.evals["score"]
            best_fn = candidate_fn

        return dataset_eval

    runs: list[ExperimentResult[Params, Response]] = await amap(
        do_evaluate,
        candidates,
        concurrency=concurrency,
    )

    return NumericOptimizationResult(best_fn, runs)


def maximize_score[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    candidates: Iterable[LMZenbase[Params, Response]],
    experimenter: ExperimentRunner[Params, Response],
    concurrency: int = 1,
) -> NumericOptimizationResult[Params, Response]:
    return syncify(amaximize_score)(
        function,
        candidates,
        experimenter,
        concurrency,
    )


async def aminimize_loss[
    Params: dict, Response: dict
](
    function: LMFunction[Params, Response],
    candidates: Iterable[LMZenbase[Params, Response]],
    experimenter: ExperimentRunner[Params, Response],
    concurrency: int = 1,
) -> NumericOptimizationResult[Params, Response]:
    run_experiment = asyncify(experimenter)

    best_loss = float("inf")
    best_fn = None

    async def do_evaluate(zenbase: LMZenbase[Params, Response]):
        nonlocal best_loss, best_fn

        candidate_fn = function.refine(zenbase)
        dataset_eval = cast(ExperimentResult, await run_experiment(candidate_fn))

        if dataset_eval.evals["score"] < best_loss:
            best_loss = dataset_eval.evals["score"]
            best_fn = candidate_fn

        return dataset_eval

    runs: list[ExperimentResult[Params, Response]] = await amap(
        do_evaluate,
        candidates,
        concurrency=concurrency,
    )

    return NumericOptimizationResult(best_fn, runs)


def minimize_loss[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    candidates: Iterable[LMZenbase[Params, Response]],
    experimenter: ExperimentRunner[Params, Response],
    concurrency: int = 1,
) -> NumericOptimizationResult[Params, Response]:
    return syncify(aminimize_loss)(
        function,
        candidates,
        experimenter,
        concurrency,
    )
