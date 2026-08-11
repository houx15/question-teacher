import pytest

from app.reference_safety import (
    ReferenceContentSafetyError,
    ReferenceSafetyPolicy,
)
from app.schemas import ProblemInput


def test_reference_safety_allows_literals_already_public_in_problem_or_answer():
    source = ProblemInput(
        problem_text="把x=2n代入方程，求m-n。",
        reference_answer="m-n=1/2",
        reference_solution_text=(
            "把x=2n代入方程，求m-n。\n最终m-n=1/2"
        ),
    )
    policy = ReferenceSafetyPolicy.from_problem(source)

    policy.ensure_safe(
        {"explanation": "把x=2n代入方程，最后得到m-n=1/2。"}
    )


def test_reference_safety_detects_raw_only_opaque_literal_without_echoing_it():
    marker = "PRIVATE-RAW-TOKEN-83d912"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="把x=2n代入方程，求m-n。",
            reference_answer="m-n=1/2",
            reference_solution_text=marker,
        )
    )

    with pytest.raises(ReferenceContentSafetyError) as captured:
        policy.ensure_safe({"nested": [{"text": marker}]})

    assert marker not in str(captured.value)


def test_reference_safety_detects_a_long_raw_only_chinese_phrase():
    private_phrase = "这是一段只存在于参考解析里的内部批注"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="把x=2n代入方程，求m-n。",
            reference_answer="m-n=1/2",
            reference_solution_text=private_phrase,
        )
    )

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe({"summary": "讲解中出现" + private_phrase})
