from random import Random
import pytest

from zenbase.optim.labeled_few_shot import LabeledFewShot
from zenbase.types import LMDemo, deflm


lmfn = deflm(lambda x: x)
demoset = [
    LMDemo(params={}, response={"output": "a"}),
    LMDemo(params={}, response={"output": "b"}),
    LMDemo(params={}, response={"output": "c"}),
    LMDemo(params={}, response={"output": "d"}),
    LMDemo(params={}, response={"output": "e"}),
    LMDemo(params={}, response={"output": "f"}),
]


def test_demo_eq():
    assert demoset[0] != demoset[1]
    assert demoset[0] == LMDemo(params={}, response={"output": "a"})


def test_seed_idempotency(samples=5, shots=2):
    optim1 = LabeledFewShot(demoset, shots=shots, samples=samples)
    optim2 = LabeledFewShot(demoset, shots=shots, samples=samples)
    optim3 = LabeledFewShot(demoset, shots=shots, samples=samples, random=Random(41))

    set1 = optim1.candidates(lmfn)
    set2 = optim2.candidates(lmfn)
    set3 = optim3.candidates(lmfn)

    assert set1 == set2
    assert set1 != set3
    assert set2 != set3


def test_insufficient_demos():
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset, shots=len(demoset) + 1)
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset, shots=2, samples=10000)


def test_shot_count(samples=5, shots=2):
    optim = LabeledFewShot(demoset, shots=shots, samples=samples)
    candidates = optim.candidates(lmfn)

    assert all(len(c.demos) == shots for c in candidates)
    assert len(candidates) == samples
