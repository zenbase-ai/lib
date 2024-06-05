from random import Random
import pytest

from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
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


def test_seed_idempotency():
    shots = 2
    batch_size = 5

    optim1 = LabeledFewShot(demoset=demoset, shots=shots)
    optim2 = LabeledFewShot(demoset=demoset, shots=shots)
    optim3 = LabeledFewShot(demoset=demoset, shots=shots, random=Random(41))

    set1 = optim1.candidates(lmfn, batch_size)
    set2 = optim2.candidates(lmfn, batch_size)
    set3 = optim3.candidates(lmfn, batch_size)

    assert set1 == set2
    assert set1 != set3
    assert set2 != set3


def test_insufficient_demos():
    with pytest.raises(AssertionError):
        LabeledFewShot(demoset=demoset, shots=len(demoset) + 1)


def test_shot_count():
    shots = 2
    batch_size = 5

    optim = LabeledFewShot(demoset=demoset, shots=shots)
    candidates = optim.candidates(lmfn, batch_size)

    assert all(len(c.demos) == shots for c in candidates)
    assert len(candidates) == batch_size
