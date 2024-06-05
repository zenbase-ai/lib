import json
import logging

from datasets import DatasetDict
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)
import pytest
import requests

from zenbase.helpers.langchain import ZenLangSmith
from zenbase.optim.metric.labeled_few_shot import LabeledFewShot
from zenbase.types import LMRequest, deflm

TESTSET_SIZE = 5
BATCH_SIZE = 2

log = logging.getLogger(__name__)


@pytest.fixture
def optim(gsm8k_demoset: list):
    return LabeledFewShot(demoset=gsm8k_demoset, shots=3)


@pytest.fixture(scope="module")
def openai():
    return wrap_openai(AsyncOpenAI())


@pytest.fixture(scope="module")
def langsmith():
    return Client()


@pytest.fixture(scope="module")
def testset(gsm8k_dataset: DatasetDict, langsmith: Client):
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


@pytest.mark.helpers
def test_langsmith_lcel_labeled_few_shot(
    langsmith: Client,
    optim: LabeledFewShot,
    testset: list,
):
    @deflm
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
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

        print("Mathing...")
        answer = chain.invoke(request.params)
        return {"answer": answer}

    scores = []
    optim.events.on("experiment", lambda r: scores.append(r.evals["score"]))

    fn = optim.train(
        optimize_lcel,
        evaluator=ZenLangSmith.metric_evaluator(
            data=testset,
            evaluators=[score_answer],
            client=langsmith,
            max_concurrency=2,
        ),
        batch_size=BATCH_SIZE,
        epochs=1,
    )

    assert fn is not None
    assert any(scores)
    assert next(s for s in scores if s >= 0.5)


@pytest.mark.anyio
@pytest.mark.helpers
async def test_langsmith_openai_json_response_labeled_few_shot(
    langsmith: Client,
    openai: AsyncOpenAI,
    optim: LabeledFewShot,
    testset: list,
):
    @deflm
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(max=8),
        before_sleep=before_sleep_log(log, logging.WARN),
    )
    @traceable
    async def optimize_openai_json_response(request: LMRequest) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are an expert math solver. Your answer must be just the number with no separators, and nothing else. Follow the format of the examples. Think step by step. Respond with a JSON object.",
            },
        ]
        for demo in request.zenbase.demos:
            messages += [
                {"role": "user", "content": json.dumps(demo.params)},
                {"role": "assistant", "content": json.dumps(demo.response)},
            ]
        messages.append({"role": "user", "content": json.dumps(request.params)})

        print("Mathing...")
        response = await openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    scores = []
    optim.events.on("experiment", lambda r: scores.append(r.evals["score"]))

    fn = await optim.atrain(
        optimize_openai_json_response,
        evaluator=ZenLangSmith.metric_evaluator(
            data=testset,
            evaluators=[score_answer],
            client=langsmith,
            max_concurrency=2,
        ),
        batch_size=BATCH_SIZE,
        epochs=1,
    )

    assert fn is not None
    assert any(scores)
    assert next(s for s in scores if s >= 0.5)
