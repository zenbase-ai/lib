from datasets import DatasetDict
import pytest

from zenbase.types import LMDemo


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--helpers", action="store_true", help="run helpers tests")


def pytest_runtest_setup(item: pytest.Item):
    if "helpers" in item.keywords and not item.config.getoption("--helpers"):
        pytest.skip("skipping integration tests")


@pytest.fixture(scope="session", autouse=True)
def env():
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(str(Path(__file__).parent.parent / ".env.test"))


@pytest.fixture(scope="session", autouse=True)
def nest_asyncio():
    import nest_asyncio

    nest_asyncio.apply()


@pytest.fixture(scope="session", autouse=True)
def vcr_config():
    return {
        "filter_headers": ["authorization", "x-api-key"],
        "filter_query_parameters": ["api_key"],
        "cassette_library_dir": "tests/cache/cassettes",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "record_mode": "new_episodes",
    }


@pytest.fixture
def gsm8k_dataset():
    import datasets

    return datasets.load_dataset("gsm8k", "main")


@pytest.fixture
def golden_demos(gsm8k_dataset: DatasetDict) -> list[LMDemo]:
    return [
        LMDemo(
            params={"question": r["question"]},
            response={"answer": r["answer"]},
        )
        for r in gsm8k_dataset["train"].select(range(5))
    ]
