"""Versioned teaching standards used by lesson-preparation roles."""

from typing import Dict, Literal


PEDAGOGY_RUBRIC_VERSION = "0.2"

ReviewCriterionId = Literal[
    "current_emphasis_correct",
    "learner_follows_why",
    "authoritative_math_alignment",
    "authentic_reasoning_sequence",
    "must_teach_coverage",
    "interaction_no_answer_leak",
    "visual_action_alignment",
    "step_purpose_and_transition",
    "display_speech_math_alignment",
    "adaptive_explanation_depth",
    "continuous_board_structure",
    "current_step_visible",
]

NON_COMPENSABLE_CRITERIA = {
    "current_emphasis_correct": "当前强调正确：每个片段必须指出此刻真正影响下一步的条件、关系、方法或结果。",
    "learner_follows_why": "学生能跟上并理解为什么：每个必要决定必须说明为什么现在这样想、这样做，以及结果如何推动下一步。",
}

HARD_REQUIREMENT_CRITERIA = {
    "authoritative_math_alignment": "数学结论和必要条件与权威教学路线一致。",
    "authentic_reasoning_sequence": "构思、探索、执行、监控可以交替，不把解题伪装成始终线性的既定步骤。",
    "must_teach_coverage": "每个 must_teach 都有可定位的讲稿证据。",
    "interaction_no_answer_leak": "互动诊断理解，不通过选项提前泄露尚未讲授的答案。",
    "visual_action_alignment": "视觉动作只在对应语句发生时出现，并引用合法语义目标。",
    "step_purpose_and_transition": "每一步都说明此刻要解决的问题、为什么现在处理，以及结果怎样推动下一步。",
    "display_speech_math_alignment": "学生看到的数学式与口播含义一致，减号、不等号等关键运算必须自然读出。",
    "adaptive_explanation_depth": "正确回答使用简短确认；错误回答按诊断类型提供更深且有边界的讲解，再回到主线。",
    "continuous_board_structure": "板书以连续步骤累积，当前步骤、已完成摘要和临时支持内容保持同一结构。",
    "current_step_visible": "当前步骤在讲解、互动支持和恢复主线时始终可见，并由受控动作请求滚动定位。",
}

REVIEW_CRITERIA: Dict[str, str] = {
    **NON_COMPENSABLE_CRITERIA,
    **HARD_REQUIREMENT_CRITERIA,
}
NON_COMPENSABLE_GATES = tuple(NON_COMPENSABLE_CRITERIA.values())
HARD_REQUIREMENTS = tuple(HARD_REQUIREMENT_CRITERIA.values())


def rubric_payload() -> Dict[str, object]:
    """Return an independently mutable, JSON-serializable rubric snapshot."""

    return {
        "version": PEDAGOGY_RUBRIC_VERSION,
        "non_compensable_gates": list(NON_COMPENSABLE_GATES),
        "hard_requirements": list(HARD_REQUIREMENTS),
        "criteria": [
            {
                "criterion_id": criterion_id,
                "description": description,
                "non_compensable": (
                    criterion_id in NON_COMPENSABLE_CRITERIA
                ),
            }
            for criterion_id, description in REVIEW_CRITERIA.items()
        ],
    }
