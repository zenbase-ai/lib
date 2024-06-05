from dataclasses import dataclass, field
from typing import Callable, TypedDict

from zenbase.types import LMFunction


class MetricEvals(TypedDict):
    score: float


@dataclass
class CandidateMetricResult[Params: dict, Response: dict]:
    function: LMFunction[Params, Response]
    evals: MetricEvals = field(default_factory=dict)


type CandidateMetricEvaluator[Params: dict, Response: dict] = Callable[
    [LMFunction[Params, Response]],
    CandidateMetricResult[Params, Response],
]
