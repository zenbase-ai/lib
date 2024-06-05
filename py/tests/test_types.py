import pytest
from zenbase.types import LMDemo, deflm


def test_demo_eq():
    demoset = [
        LMDemo(params={}, response={"output": "a"}),
        LMDemo(params={}, response={"output": "b"}),
    ]

    # Structural inequality
    assert demoset[0] != demoset[1]
    # Structural equality
    assert demoset[0] == LMDemo(params={}, response={"output": "a"})


def test_lm_function_refine():
    fn = deflm(lambda r: r.params)
    assert fn != fn.refine()


@pytest.mark.anyio
async def test_lm_function_async():
    fn = deflm(lambda r: r.params)
    assert fn.call_sync({"answer": 42}) == await fn({"answer": 42})
