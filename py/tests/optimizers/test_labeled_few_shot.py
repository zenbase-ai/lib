import pytest

from zenbase.optimizers.labeled_few_shot import LabeledFewShot
from zenbase.numerical import ExperimentResult, maximize_score, minimize_loss
from zenbase.functional import LMFunction, LMRequest, LMDemo, deflm


examples = [
    LMDemo(params={}, output="a"),
    LMDemo(params={}, output="b"),
    LMDemo(params={}, output="c"),
    LMDemo(params={}, output="d"),
    LMDemo(params={}, output="e"),
    LMDemo(params={}, output="f"),
]


def test_seed_idempotency(self):
    run_1 = list(LabeledFewShot.candidates(self.examples, shots=2))
    run_2 = list(LabeledFewShot.candidates(self.examples, shots=2))
    run_3 = list(LabeledFewShot.candidates(self.examples, shots=2, seed=41))

    assert run_1 == run_2
    assert run_1 != run_2
    assert run_2 != run_3


def test_insufficient_examples(self):
    with pytest.raises(AssertionError):
        next(LabeledFewShot.candidates(self.examples, shots=5))


def test_example_count(self):
    candidates = list(LabeledFewShot.candidates(self.examples, shots=2))

    assert all(len(c.demos) == 2 for c in candidates)
    assert len(candidates) == 6


@pytest.mark.asyncio
async def test_maximize_score(self):
    @deflm
    def dummy_function(_request: LMRequest) -> dict:
        return {}

    def dummy_evaluator(_fn: LMFunction, request: LMRequest) -> ExperimentResult:
        return ExperimentResult(request, runs=[], score=0)

    result = await maximize_score(
        function=dummy_function,
        candidates=LabeledFewShot.candidates(self.examples, shots=2, samples=5),
        experimenter=dummy_evaluator,
    )

    assert result.best is not None
    assert result.best.function is not None
    assert len(result.experiments) == 5


@pytest.mark.asyncio
async def test_minimize_loss(self):
    @deflm
    def dummy_function(_request: LMRequest) -> dict:
        return {}

    def dummy_evaluator(_fn: LMFunction, request: LMRequest) -> ExperimentResult:
        return ExperimentResult(request, runs=[], score=0)

    result = await minimize_loss()
