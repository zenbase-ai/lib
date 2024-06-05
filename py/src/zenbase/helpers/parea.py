from typing import Callable
from parea.experiment.experiment import Experiment, ExperimentStatsSchema

from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.types import LMFunction
from zenbase.utils import random_name_gen


type PareaMetricEvals = Callable[[dict[str, float]], MetricEvals]


class ZenParea:
    @staticmethod
    def default_metric(stats: ExperimentStatsSchema) -> float:
        evals = {"score": sum(stats.avg_scores.values())}
        evals.update(stats.avg_scores)
        return evals

    @classmethod
    def metric_evaluator[
        Params: dict, Response: dict
    ](
        cls,
        *args,
        metric_evals: PareaMetricEvals = default_metric,
        **kwargs,
    ) -> CandidateMetricEvaluator:
        gen_random_name = random_name_gen(kwargs.pop("experiment_name", None))

        def run_experiment(
            function: LMFunction[Params, Response]
        ) -> CandidateMetricResult[Params, Response]:
            experiment = Experiment(func=function.call_sync, *args, **kwargs)
            experiment.run(gen_random_name())

            return CandidateMetricResult(
                function,
                evals=metric_evals(experiment.experiment_stats),
            )

        return run_experiment
