from dataclasses import asdict
import json
from typing import Callable
from parea import Parea
from parea.schemas import ExperimentStatsSchema

from zenbase.optim.metric.types import (
    MetricEvals,
    CandidateMetricResult,
    CandidateMetricEvaluator,
)
from zenbase.types import LMFunction
from zenbase.utils import random_name_gen


class ZenParea:
    type MetricEvaluator = Callable[[dict[str, float]], MetricEvals]

    @staticmethod
    def default_candidate_evals(stats: ExperimentStatsSchema) -> MetricEvals:
        return {**stats.avg_scores, "score": sum(stats.avg_scores.values())}

    @classmethod
    def metric_evaluator[
        Inputs: dict, Outputs: dict
    ](
        cls,
        *args,
        p: Parea | None = None,
        candidate_evals: MetricEvaluator = default_candidate_evals,
        **kwargs,
    ) -> CandidateMetricEvaluator:
        p = p or Parea()
        assert isinstance(p, Parea)

        base_metadata = kwargs.pop("metadata", {})
        gen_random_name = random_name_gen(kwargs.pop("name", None))

        def evaluate_candidate(
            function: LMFunction[Inputs, Outputs]
        ) -> CandidateMetricResult[Inputs, Outputs]:
            experiment = p.experiment(
                func=function.call_sync,
                *args,
                **kwargs,
                name=gen_random_name(),
                metadata={
                    **base_metadata,
                    "zenbase": json.dumps(asdict(function.zenbase)),
                },
            )

            experiment.run()
            assert experiment.experiment_stats, "failed to run experiment"

            return CandidateMetricResult(
                function,
                evals=candidate_evals(experiment.experiment_stats),
            )

        return evaluate_candidate
