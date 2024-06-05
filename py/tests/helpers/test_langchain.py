import json

from datasets import DatasetDict
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
import pytest
import requests


from zenbase.helpers.langchain import ZenLangSmith
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.types import LMRequest, deflm
from zenbase.optim.metric.types import maximize_score

TESTSET_SIZE = 5
SAMPLE_SIZE = 2


@pytest.fixture
def openai():
    return wrap_openai(AsyncOpenAI())


@pytest.fixture
def langsmith():
    return Client()


@pytest.fixture
def test_examples(gsm8k_dataset: DatasetDict, langsmith: Client):
    try:
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 404:
            raise
        dataset = langsmith.create_dataset("gsm8k-test-examples")
        examples = gsm8k_dataset["test"].select(TESTSET_SIZE)
        langsmith.create_examples(
            inputs=[{"question": e["question"]} for e in examples],
            outputs=[{"answer": e["answer"]} for e in examples],
            dataset_id=dataset.id,
        )
        return list(langsmith.list_examples(dataset_name="gsm8k-test-examples"))


def score_answer(run: Run, example: Example) -> bool:
    return {
        "key": "correctness",
        "score": int(
            run.outputs["answer"].split("#### ")[-1]
            == example.outputs["answer"].split("#### ")[-1]
        ),
    }


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.helpers
async def test_lcel_labeled_few_shot(
    langsmith: Client,
    test_examples: list,
    golden_demos: list,
):
    @deflm
    @traceable
    def optimize_lcel(request: LMRequest):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        messages = [
            (
                "system",
                "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples.",
            )
        ]
        for demo in request.zenbase.demos:
            messages += [
                ("user", demo.params["question"]),
                ("assistant", demo.response["answer"]),
            ]

        messages.append(("user", "{question}"))

        chain = (
            ChatPromptTemplate.from_messages(messages)
            | ChatOpenAI(model="gpt-3.5-turbo")
            | StrOutputParser()
        )

        answer = chain.invoke(request.params)
        return {"answer": answer}

    result = maximize_score(
        function=optimize_lcel,
        optimizer=LabeledFewShot(golden_demos, samples=SAMPLE_SIZE),
        evaluator=ZenLangSmith.metric_evaluator(
            test_examples,
            evaluators=[score_answer],
            client=langsmith,
        ),
    )

    assert result.function is not None
    assert len(result.experiments) == SAMPLE_SIZE


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.helpers
async def test_openai_json_response_labeled_few_shot(
    langsmith: Client,
    golden_demos: list,
    test_examples: list,
    openai: AsyncOpenAI,
):
    @deflm
    @traceable
    async def optimize_openai_json_response(request: LMRequest) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples. Respond with a JSON object.",
            },
        ]
        for example in request.zenbase.demos:
            messages += [
                {"role": "user", "content": json.dumps(example["inputs"])},
                {"role": "assistant", "content": json.dumps(example["outputs"])},
            ]
        messages.append({"role": "user", "content": json.dumps(request.params)})

        print("Mathing...")
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    result = maximize_score(
        function=optimize_openai_json_response,
        optimizer=LabeledFewShot(golden_demos, samples=SAMPLE_SIZE),
        evaluator=ZenLangSmith.metric_evaluator(
            test_examples,
            evaluators=[score_answer],
            client=langsmith,
        ),
    )

    assert result.function is not None
    assert len(result.experiments) == SAMPLE_SIZE
