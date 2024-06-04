from datasets import DatasetDict
from openai import AsyncOpenAI
import pandas as pd
import pytest

from zenbase.omakase.arize import PhoenixZen
from zenbase.optimizers.labeled_few_shot import LabeledFewShot
from zenbase.functional import LMDemo, LMRequest
from zenbase.numerical import maximize_score

TEST_SIZE = 5
SAMPLE_SIZE = 2


@pytest.fixture
def openai():
    return AsyncOpenAI()


@pytest.fixture
def golden_demos_df(golden_demos: list[LMDemo]) -> pd.DataFrame:
    # TODO
    return pd.DataFrame([])


@pytest.fixture
def test_examples_df(gsm8k_dataset: DatasetDict) -> pd.DataFrame:
    # TODO
    return pd.DataFrame([])


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_arize_phoenix_labeled_few_shot(
    test_examples_df: pd.DataFrame,
    golden_demos_df: pd.DataFrame,
    openai: AsyncOpenAI,
):
    async def function(question: str, inputs: LMRequest, return_inputs: bool = False):
        if return_inputs:
            return inputs

        few_shot_examples = inputs["examples"]
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",
            },
        ]
        for example in few_shot_examples:
            messages += [
                {"role": "user", "content": example["inputs"]["question"]},
                {"role": "assistant", "content": example["outputs"]["answer"]},
            ]
        messages.append({"role": "user", "content": question})

        print("Mathing...")
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        return response.choices[0].message.content

    def evaluator() -> pd.DataFrame: ...

    optimized_function, run = await LabeledFewShot.maximize_score(
        function,
        demos=PhoenixZen.demos(golden_demos_df),
        evaluator=PhoenixZen.metric_evaluator(
            evaluator,
            testset=test_examples_df,
            concurrency=20,
        ),
        samples=SAMPLE_SIZE,
    )
