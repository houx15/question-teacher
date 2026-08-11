"""Versioned teaching standards used by lesson-preparation roles."""

from typing import Dict, List, Union


PEDAGOGY_RUBRIC_VERSION = "0.1"

NON_COMPENSABLE_GATES = (
    "当前强调正确：每个片段必须指出此刻真正影响下一步的条件、关系、方法或结果。",
    "学生能跟上并理解为什么：每个必要决定必须说明为什么现在这样想、这样做，以及结果如何推动下一步。",
)

HARD_REQUIREMENTS = (
    "数学结论和必要条件与权威教学路线一致。",
    "构思、探索、执行、监控可以交替，不把解题伪装成始终线性的既定步骤。",
    "每个 must_teach 都有可定位的讲稿证据。",
    "互动诊断理解，不通过选项提前泄露尚未讲授的答案。",
    "视觉动作只在对应语句发生时出现，并引用合法语义目标。",
)


def rubric_payload() -> Dict[str, Union[str, List[str]]]:
    """Return an independently mutable, JSON-serializable rubric snapshot."""

    return {
        "version": PEDAGOGY_RUBRIC_VERSION,
        "non_compensable_gates": list(NON_COMPENSABLE_GATES),
        "hard_requirements": list(HARD_REQUIREMENTS),
    }
