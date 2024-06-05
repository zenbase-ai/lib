from dataclasses import dataclass, field
from typing import Callable, NamedTuple, TypedDict

from zenbase.types import LMFunction, LMCall


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
