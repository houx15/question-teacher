import asyncio
import copy
import json

import pytest

from app.generation import LessonGenerationService, LessonQualityError
from app.math_engine import MathEngine
from app.prompts import DIRECTOR_SYSTEM, REVIEWER_SYSTEM, REVISION_SYSTEM
from app.schemas import ProblemInput


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def problem(required_method="factor"):
    return ProblemInput(
        problem_text="用指定方法解方程：x^2-5x+6=0",
        reference_answer="x=2 或 x=3",
        required_method=required_method,
    )


def valid_draft():
    return {
        "title": "把二次式拆成两个一次因式",
        "learning_goal": "理解因式分解如何把二次方程变成两个一次方程。",
        "opening": "先观察原式：哪两个数相乘是 6，相加是 -5？",
        "method_rationale": "首项系数为 1，常数 6 可拆成两个整数的乘积。",
        "math_steps": [
            {
                "purpose": "因式分解",
                "operation": "factor",
                "operands": [],
                "state_before": ["x^2-5x+6=0"],
                "state_after": ["(x-2)(x-3)=0"],
                "reason": "两个数相乘为 6、相加为 -5。",
            }
        ],
        "moments": [
            {
                "purpose": "寻找因数关系",
                "narration": "先自己找一找：哪两个数满足乘积与和的条件？",
                "board_actions": [
                    {"type": "focus", "target": "constant_and_linear_terms"}
                ],
                "layer": "interaction",
                "interaction": {
                    "interaction_id": "find-factor-pair",
                    "kind": "free_text",
                    "prompt": "写出这两个数。",
                    "expected_answer": "-2 和 -3",
                    "hints": ["先列出 6 的整数因数对。"],
                    "explanation_after_correct": "这组数同时满足乘积和相加条件。",
                },
            },
            {
                "purpose": "写出因式分解",
                "narration": "用刚才找到的两个数，把二次式写成两个一次因式。",
                "board_actions": [
                    {
                        "type": "transform",
                        "target": "equation",
                        "content": "(x-2)(x-3)=0",
                    }
                ],
            },
        ],
        "summary": "因式分解后，让每个一次因式分别等于零。",
        "transfer_item": {
            "problem_text": "用因式分解法解方程：x^2-7x+12=0",
            "expected_answer": "x=3 或 x=4",
            "method_signal": "寻找乘积为 12、和为 -7 的两个数。",
        },
    }


def approved_review():
    return {
        "status": "approved",
        "overall_assessment": "主线完整，互动位于关键认知转折点。",
        "must_fix": [],
        "evidence": ["学生先寻找因数关系，再看到因式分解。"],
    }


def revision_review():
    return {
        "status": "revision_required",
        "overall_assessment": "关键理由还不够清楚。",
        "must_fix": ["解释两个数为什么同时满足乘积与和的条件。"],
        "evidence": ["当前开场只提出问题，没有连接到因式分解。"],
    }


def test_approved_draft_is_compiled_without_rewrite():
    client = FakeClient([valid_draft(), approved_review()])
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(service.generate(problem()))

    assert len(client.calls) == 2
    assert client.calls[0][0] == DIRECTOR_SYSTEM
    assert client.calls[1][0] == REVIEWER_SYSTEM
    assert lesson.validation_report == {
        "math_status": "verified",
        "review_status": "approved",
        "revision_count": 0,
        "independent_solutions": ["2", "3"],
        "review_assessment": "主线完整，互动位于关键认知转折点。",
    }
    director_payload = json.loads(client.calls[0][1])
    assert director_payload["problem"]["required_method"] == "factor"
    assert director_payload["independent_solutions"] == ["2", "3"]
    assert "lesson_schema" in director_payload


def test_revision_required_returns_whole_lesson_to_director():
    revised = valid_draft()
    revised["opening"] = "先把乘积与和的条件连起来，再选择因数对。"
    client = FakeClient(
        [valid_draft(), revision_review(), revised, approved_review()]
    )
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(service.generate(problem()))

    assert len(client.calls) == 4
    assert client.calls[2][0] == REVISION_SYSTEM
    revision_payload = json.loads(client.calls[2][1])
    assert revision_payload["current_whole_lesson"]["title"] == valid_draft()["title"]
    assert revision_payload["review"]["must_fix"] == revision_review()["must_fix"]
    assert lesson.validation_report["revision_count"] == 1


def test_two_revisions_are_the_maximum():
    draft = valid_draft()
    client = FakeClient(
        [
            draft,
            revision_review(),
            copy.deepcopy(draft),
            revision_review(),
            copy.deepcopy(draft),
            revision_review(),
        ]
    )
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="两轮修订"):
        asyncio.run(service.generate(problem()))

    assert len(client.calls) == 6


def test_original_problem_is_validated_before_model_request():
    client = FakeClient([])
    service = LessonGenerationService(client, MathEngine())
    invalid_problem = ProblemInput(
        problem_text="x^2-5x+6=0",
        reference_answer="x=100",
    )

    with pytest.raises(LessonQualityError, match="题目"):
        asyncio.run(service.generate(invalid_problem))

    assert client.calls == []


def test_invalid_math_step_stops_before_review_with_safe_error():
    draft = valid_draft()
    draft["math_steps"][0]["state_after"] = ["(x-1)(x-6)=0"]
    client = FakeClient([draft])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError) as exc_info:
        asyncio.run(service.generate(problem()))

    assert "数学步骤" in str(exc_info.value)
    assert "(x-1)(x-6)" not in str(exc_info.value)
    assert len(client.calls) == 1


def test_math_route_rejects_unrelated_first_step_with_safe_error():
    draft = valid_draft()
    draft["math_steps"][0]["state_before"] = ["x^2-9=0"]
    draft["math_steps"][0]["state_after"] = ["(x-3)(x+3)=0"]
    client = FakeClient([draft])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError) as exc_info:
        asyncio.run(service.generate(problem()))

    assert "数学路线" in str(exc_info.value)
    assert "x^2-9" not in str(exc_info.value)
    assert len(client.calls) == 1


def test_math_route_rejects_disconnected_consecutive_steps():
    draft = valid_draft()
    draft["math_steps"].append(
        {
            "purpose": "插入无关但同解集的变形",
            "operation": "multiply_both_sides",
            "operands": ["2"],
            "state_before": ["x^2-5x+6=0"],
            "state_after": ["2*x^2-10*x+12=0"],
            "reason": "等式两边同时乘二。",
        }
    )
    client = FakeClient([draft])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="数学路线"):
        asyncio.run(service.generate(problem()))

    assert len(client.calls) == 1


def test_math_route_rejects_final_solution_mismatch():
    class PermissiveStepMathEngine(MathEngine):
        def validate_step(self, step):
            return None

    draft = valid_draft()
    draft["math_steps"][0]["state_after"] = ["x=99"]
    client = FakeClient([draft])
    service = LessonGenerationService(
        client,
        PermissiveStepMathEngine(),
    )

    with pytest.raises(LessonQualityError) as exc_info:
        asyncio.run(service.generate(problem()))

    assert "数学路线" in str(exc_info.value)
    assert "x=99" not in str(exc_info.value)
    assert len(client.calls) == 1


def test_math_route_accepts_contiguous_normalized_steps():
    draft = valid_draft()
    draft["math_steps"] = [
        {
            "purpose": "两边减六",
            "operation": "subtract_both_sides",
            "operands": ["6"],
            "state_before": ["x^2 - 5*x + 6 = 0"],
            "state_after": ["x^2-5x=-6"],
            "reason": "等式两边同时减六。",
        },
        {
            "purpose": "两边加六",
            "operation": "add_both_sides",
            "operands": ["6"],
            "state_before": ["x² − 5x = -6"],
            "state_after": ["x^2-5x+6=0"],
            "reason": "等式两边同时加六。",
        },
        valid_draft()["math_steps"][0],
    ]
    client = FakeClient([draft, approved_review()])
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(service.generate(problem()))

    assert lesson.validation_report["math_status"] == "verified"
    assert len(client.calls) == 2


def test_math_route_preserves_valid_multi_branch_final_state():
    draft = valid_draft()
    draft["math_steps"] = [
        {
            "purpose": "使用求根公式",
            "operation": "quadratic_formula",
            "operands": [],
            "state_before": ["x^2-5x+6=0"],
            "state_after": ["x=3", "x=2"],
            "reason": "代入求根公式并分别计算两个根。",
        }
    ]
    client = FakeClient([draft, approved_review()])
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(
        service.generate(problem("quadratic_formula"))
    )

    assert lesson.validation_report["math_status"] == "verified"


def test_math_route_preserves_valid_no_real_solution_state():
    no_real_problem = ProblemInput(
        problem_text="用求根公式解方程：x^2+1=0",
        reference_answer="无实数解",
        required_method="quadratic_formula",
    )
    draft = valid_draft()
    draft["math_steps"] = [
        {
            "purpose": "使用求根公式",
            "operation": "quadratic_formula",
            "operands": [],
            "state_before": ["x^2+1=0"],
            "state_after": ["无实数解"],
            "reason": "判别式小于零，所以没有实数根。",
        }
    ]
    draft["transfer_item"] = {
        "problem_text": "用求根公式解方程：x^2+4=0",
        "expected_answer": "无实数解",
        "method_signal": "先判断判别式的符号。",
    }
    client = FakeClient([draft, approved_review()])
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(service.generate(no_real_problem))

    assert lesson.validation_report["independent_solutions"] == []


def test_required_method_must_be_used_as_an_operation():
    client = FakeClient([valid_draft()])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="指定方法"):
        asyncio.run(service.generate(problem("complete_the_square")))

    assert len(client.calls) == 1


def test_invalid_transfer_item_stops_before_review():
    draft = valid_draft()
    draft["transfer_item"]["expected_answer"] = "x=30 或 x=40"
    client = FakeClient([draft])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="近迁移题"):
        asyncio.run(service.generate(problem()))

    assert len(client.calls) == 1


def test_draft_requires_at_least_one_interaction():
    draft = valid_draft()
    for moment in draft["moments"]:
        moment.pop("interaction", None)
        moment["layer"] = "base"
    client = FakeClient([draft])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="学生互动"):
        asyncio.run(service.generate(problem()))


def test_invalid_director_schema_is_reported_without_model_payload():
    private_payload = {
        "title": "private-model-output",
        "math_steps": "not-a-list",
    }
    client = FakeClient([private_payload])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError) as exc_info:
        asyncio.run(service.generate(problem()))

    assert "讲解结构无效" in str(exc_info.value)
    assert "private-model-output" not in str(exc_info.value)


def test_sync_stage_callback_receives_generation_stages_in_order():
    stages = []
    client = FakeClient([valid_draft(), approved_review()])
    service = LessonGenerationService(client, MathEngine())

    asyncio.run(service.generate(problem(), on_stage=stages.append))

    assert stages == [
        "正在验证数学路线",
        "正在设计完整讲解",
        "正在进行整篇审稿",
        "正在编译课堂",
    ]


def test_async_stage_callback_is_awaited():
    stages = []

    async def on_stage(stage):
        await asyncio.sleep(0)
        stages.append(stage)

    client = FakeClient([valid_draft(), approved_review()])
    service = LessonGenerationService(client, MathEngine())

    asyncio.run(service.generate(problem(), on_stage=on_stage))

    assert stages[-1] == "正在编译课堂"
    assert len(stages) == 4


def test_prompt_contracts_state_teaching_and_output_constraints():
    assert "完整" in DIRECTOR_SYSTEM
    assert "一个主要认知动作" in DIRECTOR_SYSTEM
    assert "最多 90 个字符" in DIRECTOR_SYSTEM
    assert "不得" in DIRECTOR_SYSTEM and "泄露" in DIRECTOR_SYSTEM
    assert "exactly one operand" in DIRECTOR_SYSTEM
    assert "write" in DIRECTOR_SYSTEM and "transform" in DIRECTOR_SYSTEM
    assert "整节课" in REVIEWER_SYSTEM
    assert "完整 LessonDraft JSON" in REVISION_SYSTEM
