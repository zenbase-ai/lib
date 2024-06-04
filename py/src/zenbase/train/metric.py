from dataclasses import dataclass, field
from typing import Callable, Iterable, NamedTuple, TypedDict, cast

from zenbase.optim.abc import LMOptim
from zenbase.types import LMFunction, LMCall, LMZenbase
from zenbase.utils import alist, amap, asyncify, syncify


class MetricEvals(TypedDict):
    score: float


class MetricExperimentRun(NamedTuple):
    call: LMCall
    evals: MetricEvals


@dataclass
class MetricExperimentResult[Params: dict, Response: dict]:
    function: LMFunction[Params, Response]
    evals: MetricEvals = field(default_factory=dict)


type MetricExperimentEvaluator[Params: dict, Response: dict] = Callable[
    [LMFunction[Params, Response]],
    MetricExperimentResult[Params, Response],
]


@dataclass
class MetricTrainingResult[Params: dict, Response: dict]:
    function: LMFunction[Params, Response]
    experiments: list[MetricExperimentResult[Params, Response]]


async def amaximize_score[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    optimizer: LMOptim[Params, Response],
    evaluator: MetricExperimentEvaluator[Params, Response],
    epochs: int = 1,
) -> MetricTrainingResult[Params, Response]:
    run_experiment = asyncify(evaluator)

    score = float("-inf")
    best = function

    async def do_experiment(zenbase: LMZenbase):
        nonlocal score, best

        candidate_fn = function.refine(zenbase)
        dataset_eval = cast(MetricExperimentResult, await run_experiment(candidate_fn))

        if dataset_eval.evals["score"] > score:
            score = dataset_eval.evals["score"]
            best = candidate_fn

        return dataset_eval

    runs: list[MetricExperimentResult[Params, Response]] = []
    for _ in range(epochs):
        candidates = await alist(optimizer.acandidates(best))
        runs += await amap(do_experiment, candidates, concurrency=1)

    return MetricTrainingResult(best, runs)


def maximize_score[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    optimizer: LMOptim[Params, Response],
    evaluator: MetricExperimentEvaluator[Params, Response],
    epochs: int = 1,
) -> MetricTrainingResult[Params, Response]:
    return syncify(amaximize_score)(function, optimizer, evaluator, epochs)


async def aminimize_loss[
    Params: dict, Response: dict
](
    function: LMFunction[Params, Response],
    optimizer: LMOptim[Params, Response],
    evaluator: MetricExperimentEvaluator[Params, Response],
    epochs: int = 1,
) -> MetricTrainingResult[Params, Response]:
    run_experiment = asyncify(evaluator)

    loss = float("inf")
    best = function

    async def do_experiment(zenbase: LMZenbase):
        nonlocal loss, best

        candidate_fn = function.refine(zenbase)
        dataset_eval = cast(MetricExperimentResult, await run_experiment(candidate_fn))

        if dataset_eval.evals["score"] < loss:
            loss = dataset_eval.evals["score"]
            best = candidate_fn

        return dataset_eval

    runs: list[MetricExperimentResult[Params, Response]] = []
    for _ in range(epochs):
        candidates = await alist(optimizer.acandidates(best))
        runs += await amap(do_experiment, candidates, concurrency=1)

    return MetricTrainingResult(best, runs)


def minimize_loss[
    Params: dict,
    Response: dict,
](
    function: LMFunction[Params, Response],
    optimizer: LMOptim[Params, Response],
    evaluator: MetricExperimentEvaluator[Params, Response],
    epochs: int = 1,
) -> MetricTrainingResult[Params, Response]:
    return syncify(aminimize_loss)(function, optimizer, evaluator, epochs)
