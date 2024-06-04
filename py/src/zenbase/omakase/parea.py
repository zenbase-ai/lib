from typing import Iterable
from parea import Experiment
from parea.helpers import gen_random_name

from zenbase.functional import LMCallable, LMFunction, LMZenbase, deflm
from zenbase.numerical import ExperimentResult, ExperimentRunner, maximize_score


class PareaZen:
    @staticmethod
    def experiment_numeric[
        Params: dict, Response: dict
    ](*args, **kwargs) -> ExperimentRunner:
        experiment_prefix = "zenbase"
        if name := kwargs.pop("experiment_name", None):
            experiment_prefix += f"-{name}"

        def run_experiment(
            function: LMFunction[Params, Response]
        ) -> ExperimentResult[Params, Response]:
            experiment = Experiment(func=function.call_sync, *args, **kwargs)
            experiment.run(f"{experiment_prefix}-{gen_random_name()}")
            return ExperimentResult(function, experiment.experiment_stats.avg_scores)

        return run_experiment

    @classmethod
    def maximize_score[
        Params: dict, Response: dict
    ](
        cls,
        fn: LMCallable[Params, Response] | LMFunction[Params, Response],
        candidates: Iterable[LMZenbase[Params, Response]],
        *args,
        **kwargs,
    ) -> ExperimentResult[Params, Response]:
        return maximize_score(
            deflm(fn), candidates, cls.experiment_numeric(*args, **kwargs)
        )
