from random import Random, random
import pytest

from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.optim.metric.types import CandidateMetricResult
from zenbase.types import LMDemo, LMFunction, LMRequest, deflm


lmfn = deflm(lambda x: x)


demoset = [
    LMDemo(params={}, response={"output": "a"}),
    LMDemo(params={}, response={"output": "b"}),
    LMDemo(params={}, response={"output": "c"}),
    LMDemo(params={}, response={"output": "d"}),
    LMDemo(params={}, response={"output": "e"}),
    LMDemo(params={}, response={"output": "f"}),
]


def test_invalid_shots():
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset=demoset, shots=0)
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset=demoset, shots=len(demoset) + 1)


def test_idempotency():
    shots = 2
    batch_size = 5

    optim1 = LabeledFewShot(demoset=demoset, shots=shots)
    optim2 = LabeledFewShot(demoset=demoset, shots=shots)
    optim3 = LabeledFewShot(demoset=demoset, shots=shots, random=Random(41))

    set1 = list(optim1.candidates(lmfn, batch_size))
    set2 = list(optim2.candidates(lmfn, batch_size))
    set3 = list(optim3.candidates(lmfn, batch_size))

    assert set1 == set2
    assert set1 != set3
    assert set2 != set3


@pytest.fixture
def optim():
    return LabeledFewShot(demoset=demoset, shots=2)


def test_candidate_generation(optim: LabeledFewShot):
    batch_size = 5

    candidates = list(optim.candidates(lmfn, batch_size))

    assert all(len(c.demos) == optim.shots for c in candidates)
    assert len(candidates) == batch_size


@deflm
def dummy_lmfn(_: LMRequest):
    return {"answer": 42}


def dummy_evalfn(fn: LMFunction):
    return CandidateMetricResult(fn, {"score": random()})


def test_training(optim: LabeledFewShot):
    # Watch epoch runs for testing
    candidate_results: list[CandidateMetricResult] = []
    optim.events.on("epoch", lambda r: candidate_results.append(r))

    # Train the dummy function
    trained_lmfn = optim.train(
        dummy_lmfn,
        dummy_evalfn,
        epochs=1,
        concurrency=1,
    )

    # Check that the best function is returned
    best_function = max(candidate_results, key=lambda r: r.evals["score"]).function
    assert trained_lmfn == best_function

    for demo in trained_lmfn.zenbase.demos:
        assert demo in demoset


@pytest.mark.anyio
async def test_async_training(optim: LabeledFewShot):
    # Watch epoch runs for testing
    candidate_results: list[CandidateMetricResult] = []
    optim.events.on("epoch", lambda r: candidate_results.append(r))

    # Train the dummy function
    trained_dummy_lmfn = await optim.atrain(
        dummy_lmfn,
        dummy_evalfn,
        epochs=1,
        concurrency=1,
    )

    # Check that the best function is returned
    best_function = max(candidate_results, key=lambda r: r.evals["score"]).function
    assert trained_dummy_lmfn == best_function
