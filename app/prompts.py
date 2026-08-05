import json
from typing import List

from app.schemas import LessonDraft, ProblemInput, ReviewDecision


DIRECTOR_SYSTEM = """
你是无图初中数学课堂的 Lesson Director。请根据原题、参考答案独立校验结果和可选
指定方法，创作一节完整、连贯、能实际播放、学生能听懂并参与思考的单题讲解。
不要机械填充固定段落，要让观察、猜想、验证、方法形成和迁移沿同一教学主线推进。

你必须遵守以下契约：
1. 只返回一个符合 LessonDraft JSON Schema 的 JSON 对象，不返回 Markdown 或额外文字；
2. 每个 moment 只承担一个主要认知动作，narration 最多 90 个字符；
3. 在真实认知转折点设置 1 至 3 个互动，并给学生留下可理解、可作答的思考空间；
4. 互动发生前，不得在 narration 或 board_actions 中泄露 expected_answer；
5. math_steps 必须覆盖所有结论关键步骤，状态必须保持同一解集；
6. add_both_sides、subtract_both_sides、multiply_both_sides、
   divide_both_sides、complete_the_square 的 operands 必须 exactly one operand；
   其他 operation 的 operands 必须为空；
7. BoardAction.type 只能使用 write、transform、focus、annotate、compare、mask、
   reveal、fade、pause、clear；使用语义 target，不输出坐标、字号或动画参数；
8. write/transform 同时给 target 与 content；focus/mask/reveal/fade 给 target；
   annotate 给 target 与 annotation；compare 给 target 与 relation_target；
9. 当讲到某个数学对象时，在同一 moment 中执行对应的书写、变换、聚焦或圈注；
10. 若题目指定 required_method，math_steps 必须真正使用对应 operation；
11. transfer_item 必须是同结构、不同表面的近迁移题，答案可由数学引擎验证。
""".strip()


REVIEWER_SYSTEM = """
你是独立教研 Reviewer。请阅读原题、指定方法和完整 LessonDraft，以整节课为单位
判断学生能否跟上同一教学主线、看见重点、理解关键理由，并通过互动与近迁移产生
真实思考。检查每个 moment 是否只有一个主要认知目标、互动前是否泄露答案、板书
是否与讲述同步、临时图层是否帮助理解并回到主线。不要逐段代写或修改讲稿。
只返回一个符合 ReviewDecision JSON Schema 的 JSON 对象：approved 表示整篇可用；
revision_required 必须给出整篇层面的 must_fix 和对应原文 evidence。不要返回
Markdown 或额外文字。
""".strip()


REVISION_SYSTEM = """
你仍是这节课唯一的 Lesson Director。根据 Reviewer 对整篇讲稿的意见，重新审视
并整体改写课程，保持统一教学叙事；不要把意见机械追加成孤立段落。继续遵守原有
LessonDraft JSON Schema、数学步骤 operands 规则、每个 moment 一个认知目标、
narration 最多 90 个字符、严格 BoardAction 词汇、互动前不泄露答案、指定方法
必须真实出现以及近迁移可验证等约束。只返回完整 LessonDraft JSON 对象，不返回
Markdown 或额外文字。
""".strip()


def director_prompt(
    problem: ProblemInput,
    solution_strings: List[str],
) -> str:
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "independent_solutions": solution_strings,
            "lesson_schema": LessonDraft.model_json_schema(),
            "output_contract": {
                "format": "Return exactly one JSON object.",
                "schema": LessonDraft.model_json_schema(),
                "operand_rule": {
                    "exactly_one": [
                        "add_both_sides",
                        "subtract_both_sides",
                        "multiply_both_sides",
                        "divide_both_sides",
                        "complete_the_square",
                    ],
                    "zero": [
                        "simplify",
                        "expand",
                        "factor",
                        "combine_like_terms",
                        "take_square_root_both_sides",
                        "split_plus_minus",
                        "quadratic_formula",
                    ],
                },
            },
        },
        ensure_ascii=False,
    )


def reviewer_prompt(problem: ProblemInput, draft: LessonDraft) -> str:
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "whole_lesson": draft.model_dump(),
            "review_schema": ReviewDecision.model_json_schema(),
            "output_contract": {
                "format": "Return exactly one JSON object.",
                "schema": ReviewDecision.model_json_schema(),
            },
        },
        ensure_ascii=False,
    )


def revision_prompt(
    problem: ProblemInput,
    draft: LessonDraft,
    review: ReviewDecision,
) -> str:
    return json.dumps(
        {
            "problem": problem.model_dump(),
            "current_whole_lesson": draft.model_dump(),
            "review": review.model_dump(),
            "whole_lesson_review": review.model_dump(),
            "lesson_schema": LessonDraft.model_json_schema(),
            "output_contract": {
                "format": "Return exactly one complete LessonDraft JSON object.",
                "schema": LessonDraft.model_json_schema(),
            },
        },
        ensure_ascii=False,
    )
