from dataclasses import dataclass
import pytest

from zenbase import (
    deflm,
    LMFunction,
    LMRequest,
    amaximize_score,
    aminimize_loss,
    maximize_score,
    minimize_loss,
)
from zenbase.optim.abc import LMOptim
from zenbase.train.metric import MetricExperimentResult
from zenbase.types import LMZenbase


@deflm
def dummy_lmfn(req: LMRequest):
    return int(req.zenbase.instructions[0])


def dummy_evalfn(fn: LMFunction):
    score = fn.call_sync({})
    return MetricExperimentResult(fn, {"score": score})


async def adummy_evalfn(fn: LMFunction):
    score = await fn({})
    return MetricExperimentResult(fn, {"score": score})


@dataclass
class DummyLMOptim(LMOptim):
    min: int
    max: int

    async def acandidates(self, fn: LMFunction):
        yield LMZenbase(instructions=[str(self.max)])
        yield LMZenbase(instructions=["12"])
        yield LMZenbase(instructions=[str(self.min)])
        yield LMZenbase(instructions=["22"])


def test_maximize_score(min_score=0, max_score=42):
    result = maximize_score(
        dummy_lmfn, DummyLMOptim(min_score, max_score), dummy_evalfn
    )
    assert result.function.zenbase.instructions == [str(max_score)]

    winner = max(result.experiments, key=lambda e: e.evals["score"])
    assert winner.function == result.function
    assert winner.evals["score"] == max_score


def test_minimize_loss(min_loss=0, max_loss=42):
    result = minimize_loss(dummy_lmfn, DummyLMOptim(min_loss, max_loss), dummy_evalfn)
    assert result.function.zenbase.instructions == [str(min_loss)]

    winner = min(result.experiments, key=lambda e: e.evals["score"])
    assert winner.function == result.function
    assert winner.evals["score"] == min_loss


@pytest.mark.asyncio
async def test_amaximize_score(min_score=0, max_score=42):
    result = await amaximize_score(
        dummy_lmfn, DummyLMOptim(min_score, max_score), adummy_evalfn
    )
    assert result.function.zenbase.instructions == [str(max_score)]

    winner = max(result.experiments, key=lambda e: e.evals["score"])
    assert winner.function == result.function
    assert winner.evals["score"] == max_score


@pytest.mark.asyncio
async def test_aminimize_loss(min_loss=0, max_loss=42):
    result = await aminimize_loss(
        dummy_lmfn, DummyLMOptim(min_loss, max_loss), adummy_evalfn
    )
    assert result.function.zenbase.instructions == [str(min_loss)]

    winner = min(result.experiments, key=lambda e: e.evals["score"])
    assert winner.function == result.function
    assert winner.evals["score"] == min_loss
