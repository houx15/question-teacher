import hashlib

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


def test_reference_safety_detects_a_partial_chinese_prose_leak():
    raw = "先判断根与参数的特殊联系，再决定如何整理目标关系"
    partial = "根与参数的特殊联系"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text=raw,
        )
    )

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe({"summary": "请观察" + partial})


def test_reference_safety_has_no_silent_candidate_cap():
    private_lines = [
        hashlib.sha256(str(index).encode("ascii")).hexdigest()[:16]
        for index in range(300)
    ]
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text="\n".join(private_lines),
        )
    )

    for index in (257, 299):
        with pytest.raises(ReferenceContentSafetyError):
            policy.ensure_safe({"summary": private_lines[index]})


@pytest.mark.parametrize(
    ("private_text", "blocked"),
    [
        ("甲乙丙丁戊己庚", False),
        ("甲乙丙丁戊己庚辛", True),
    ],
)
def test_reference_safety_uses_an_explicit_eight_character_boundary(
    private_text,
    blocked,
):
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text=private_text,
        )
    )

    if blocked:
        with pytest.raises(ReferenceContentSafetyError):
            policy.ensure_safe({"summary": private_text})
    else:
        policy.ensure_safe({"summary": private_text})


def test_reference_safety_blocks_a_long_opaque_token():
    token = "opaque-private-token-7b91fe02"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text=token,
        )
    )

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe({"summary": "prefix " + token + " suffix"})

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe({"summary": "private-token-7b91fe02"})


def test_reference_safety_blocks_partial_english_prose():
    raw = "notice the hidden relationship before choosing the next operation"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="Solve the equation for the requested value.",
            reference_answer="m-n=1/2",
            reference_solution_text=raw,
        )
    )

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe(
            {"summary": "The hidden relationship should be noticed."}
        )


def test_reference_safety_does_not_trust_fake_math_delimiters():
    disguised_prose = "这是藏在伪公式里的内部批注"
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text="$" + disguised_prose + "$",
        )
    )

    with pytest.raises(ReferenceContentSafetyError):
        policy.ensure_safe({"summary": disguised_prose})


@pytest.mark.parametrize(
    "safe_math",
    [
        "4n^2-4mn+2n=0",
        "由条件可得4n^2-4mn+2n=0，然后继续整理",
        "123-456-789",
    ],
)
def test_reference_safety_does_not_classify_intermediate_math_as_private_prose(
    safe_math,
):
    formula = (
        safe_math
        if safe_math == "123-456-789"
        else "4n^2-4mn+2n=0"
    )
    policy = ReferenceSafetyPolicy.from_problem(
        ProblemInput(
            problem_text="已知一个根，求参数关系。",
            reference_answer="m-n=1/2",
            reference_solution_text=formula,
        )
    )

    policy.ensure_safe({"mathematical_action": safe_math})
