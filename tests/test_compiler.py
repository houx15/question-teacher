import json
from pathlib import Path
import re

import pytest

import app.compiler as compiler_module
import app.prepared_lesson_adapter as prepared_adapter
import app.schemas as schemas_module
from app.compiler import LessonCompileError, LessonCompiler
from app.problem_focus import compile_problem_focus_targets
from app.schemas import LessonDraft, NarrativeSyncCue, ProblemInput
from tests.test_generation import problem, valid_draft


def compile_lesson(compiler=None):
    return (compiler or LessonCompiler()).compile(
        problem(),
        LessonDraft.model_validate(valid_draft()),
        {"review_status": "approved"},
    )


def test_compiler_creates_ordered_beats_with_stable_navigation():
    lesson = compile_lesson(
        LessonCompiler(lesson_id_factory=lambda: "lesson-fixed")
    )

    assert lesson.lesson_id == "lesson-fixed"
    assert len(lesson.beats) == len(valid_draft()["moments"]) + 4
    assert [beat.beat_id for beat in lesson.beats] == [
        "beat-001",
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
        "beat-006",
    ]
    assert [beat.next_beat_id for beat in lesson.beats] == [
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
        "beat-006",
        None,
    ]


def test_compiler_opening_is_voice_only_and_preserves_the_problem():
    source_problem = problem()
    lesson = compile_lesson()
    opening = lesson.beats[0]

    assert opening.purpose == "进入问题"
    assert opening.narration == valid_draft()["opening"]
    assert len(opening.sync_cues) == 1
    assert opening.sync_cues[0].spoken_text == valid_draft()["opening"]
    assert opening.sync_cues[0].lead_actions == []
    assert opening.sync_cues[0].start_actions == []
    assert opening.sync_cues[0].end_actions == []
    assert opening.board_actions == []
    assert lesson.problem == source_problem


@pytest.mark.parametrize("problem_length", [500, 501, 4096])
def test_compiler_accepts_long_problem_text_without_board_duplication(
    problem_length,
):
    source_problem = ProblemInput(
        problem_text="题" * problem_length,
        reference_answer="已提供参考答案",
    )

    lesson = LessonCompiler().compile(
        source_problem,
        LessonDraft.model_validate(valid_draft()),
        {"review_status": "approved"},
    )

    assert len(lesson.problem.problem_text) == problem_length
    assert lesson.problem == source_problem
    assert lesson.beats[0].sync_cues[0].start_actions == []
    assert lesson.beats[0].board_actions == []


@pytest.mark.parametrize(
    ("student_definition", "why_it_helps", "expected_narration"),
    [
        (
            "把二次式写成两个一次因式的乘积",
            "这样能把二次方程拆成一次方程",
            "今天用因式分解法。把二次式写成两个一次因式的乘积。这样能把二次方程拆成一次方程。",
        ),
        (
            "把二次式写成两个一次因式的乘积。",
            "这样能把二次方程拆成一次方程！",
            "今天用因式分解法。把二次式写成两个一次因式的乘积。这样能把二次方程拆成一次方程！",
        ),
    ],
)
def test_compiler_introduces_method_before_first_manuscript_moment(
    student_definition,
    why_it_helps,
    expected_narration,
):
    draft = valid_draft()
    draft["method_introduction"]["student_definition"] = (
        student_definition
    )
    draft["method_introduction"]["why_it_helps"] = why_it_helps
    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )
    introduction = draft["method_introduction"]
    method_beat = lesson.beats[1]

    assert method_beat.purpose == "先认识方法"
    assert method_beat.layer == "micro_explanation"
    assert method_beat.narration.startswith(
        f"今天用{introduction['method_name']}。"
    )
    assert method_beat.narration == expected_narration
    assert introduction["student_definition"] in method_beat.narration
    assert introduction["why_it_helps"] in method_beat.narration
    assert introduction["target_form"] not in method_beat.narration
    assert r"\(" not in method_beat.narration
    assert [
        action.model_dump(exclude_none=True)
        for action in method_beat.board_actions
    ] == [
        {
            "type": "write",
            "target": "method_name",
            "content": introduction["method_name"],
        },
        {"type": "focus", "target": "method_name"},
        {
            "type": "write",
            "target": "method_target_form",
            "content": introduction["target_form"],
        },
    ]


def test_compiler_preserves_manuscript_moments_in_order():
    draft = valid_draft()
    lesson = compile_lesson()

    manuscript_beats = lesson.beats[2 : 2 + len(draft["moments"])]
    assert [beat.purpose for beat in manuscript_beats] == [
        moment["purpose"] for moment in draft["moments"]
    ]
    assert manuscript_beats[0].interaction.interaction_id == "find-factor-pair"
    assert manuscript_beats[1].board_actions[0].type == "transform"


def test_compiler_preserves_runtime_sync_cues_and_derives_legacy_fields():
    draft_payload = valid_draft()
    draft_payload["moments"][0]["sync_cues"] = [
        {
            "cue_id": "inspect-problem-cue",
            "spoken_text": "先在原题中找到这个式子。",
            "lead_actions": [
                {
                    "surface": "problem",
                    "type": "focus",
                    "target": "problem-math-001",
                }
            ],
            "start_actions": [
                {
                    "surface": "board",
                    "type": "write",
                    "target": "factor-line",
                    "content": "先找乘积与和",
                }
            ],
            "end_actions": [
                {
                    "surface": "board",
                    "type": "emphasize",
                    "target": "factor-line",
                    "emphasis_style": "highlight",
                    "persistence": "trace",
                }
            ],
        },
        {
            "cue_id": "transform-equation-cue",
            "spoken_text": "再把原式改写成两个一次因式。",
            "lead_actions": [
                {
                    "surface": "board",
                    "type": "focus",
                    "target": "factor-line",
                }
            ],
            "start_actions": [
                {
                    "surface": "board",
                    "type": "transform",
                    "target": "factor-line",
                    "content": r"\((x-2)(x-3)=0\)",
                }
            ],
            "end_actions": [
                {
                    "surface": "problem",
                    "type": "clear_focus",
                    "target": "problem-math-001",
                }
            ],
        },
    ]
    draft = LessonDraft.model_validate(draft_payload)
    source_problem = problem().model_copy(
        update={
            "problem_text": (
                r"用指定方法解方程：\(x^2-5x+6=0\)，"
                r"并说明\[x=2\]是否成立"
            )
        }
    )

    lesson = LessonCompiler().compile(
        source_problem,
        draft,
        {"review_status": "approved"},
    )

    assert [
        (target.target_id, target.math_text, target.display_mode)
        for target in lesson.problem_focus_targets
    ] == [
        ("problem-math-001", "x^2-5x+6=0", False),
        ("problem-math-002", "x=2", True),
    ]
    assert lesson.problem_focus_targets == compile_problem_focus_targets(
        source_problem.problem_text
    )

    runtime_beat = lesson.beats[2]
    source_cues = draft.moments[0].sync_cues
    assert [cue.cue_id for cue in runtime_beat.sync_cues] == [
        "inspect-problem-cue",
        "transform-equation-cue",
    ]
    assert [cue.spoken_text for cue in runtime_beat.sync_cues] == [
        cue.spoken_text for cue in source_cues
    ]
    assert runtime_beat.narration == "".join(
        cue.spoken_text for cue in source_cues
    )
    for source_cue, runtime_cue in zip(
        source_cues,
        runtime_beat.sync_cues,
    ):
        assert runtime_cue is not source_cue
        assert runtime_cue.audio_url is None
        for action_slot in (
            "lead_actions",
            "start_actions",
            "end_actions",
        ):
            source_actions = getattr(source_cue, action_slot)
            runtime_actions = getattr(runtime_cue, action_slot)
            assert [
                action.model_dump(exclude_none=True)
                for action in runtime_actions
            ] == [
                action.model_dump(exclude_none=True)
                for action in source_actions
            ]
            assert all(
                runtime_action is not source_action
                for source_action, runtime_action in zip(
                    source_actions,
                    runtime_actions,
                )
            )

    assert [
        action.model_dump(exclude_none=True)
        for action in runtime_beat.board_actions
    ] == [
        {
            "type": "write",
            "target": "factor-line",
            "content": "先找乘积与和",
        },
        {
            "type": "transform",
            "target": "factor-line",
            "content": r"\((x-2)(x-3)=0\)",
        },
    ]


def test_compiler_projects_only_supported_start_actions_to_legacy_board():
    draft_payload = valid_draft()
    draft_payload["moments"][0]["sync_cues"] = [
        {
            "cue_id": "legacy-start-projection-cue",
            "spoken_text": "按顺序展示这一步的板书变化。",
            "lead_actions": [
                {
                    "surface": "board",
                    "type": "focus",
                    "target": "projection-line",
                }
            ],
            "start_actions": [
                {
                    "surface": "board",
                    "type": "write",
                    "target": "projection-line",
                    "content": "先写出原式",
                },
                {
                    "surface": "board",
                    "type": "annotate",
                    "target": "projection-line",
                    "annotation": "underline",
                },
                {
                    "surface": "board",
                    "type": "transform",
                    "target": "projection-line",
                    "content": r"\((x-2)(x-3)=0\)",
                },
                {
                    "surface": "board",
                    "type": "fade",
                    "target": "projection-line",
                },
                {
                    "surface": "board",
                    "type": "focus",
                    "target": "projection-line",
                },
                {
                    "surface": "board",
                    "type": "emphasize",
                    "target": "projection-line",
                    "emphasis_style": "highlight",
                },
                {
                    "surface": "board",
                    "type": "reveal",
                    "target": "projection-line",
                },
                {
                    "surface": "board",
                    "type": "clear_focus",
                    "target": "projection-line",
                },
            ],
            "end_actions": [
                {
                    "surface": "board",
                    "type": "fade",
                    "target": "projection-line",
                },
                {
                    "surface": "board",
                    "type": "clear_focus",
                    "target": "projection-line",
                },
            ],
        },
        {
            "cue_id": "problem-action-stays-runtime-only-cue",
            "spoken_text": "原题上的定位只在同步动作中执行。",
            "start_actions": [
                {
                    "surface": "problem",
                    "type": "focus",
                    "target": "problem-math-001",
                }
            ],
        },
    ]
    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft_payload),
        {"review_status": "approved"},
    )

    runtime_beat = lesson.beats[2]
    assert [
        action.type
        for cue in runtime_beat.sync_cues
        for action in (
            *cue.lead_actions,
            *cue.start_actions,
            *cue.end_actions,
        )
    ] == [
        "focus",
        "write",
        "annotate",
        "transform",
        "fade",
        "focus",
        "emphasize",
        "reveal",
        "clear_focus",
        "fade",
        "clear_focus",
        "focus",
    ]
    assert [
        action.model_dump(exclude_none=True)
        for action in runtime_beat.board_actions
    ] == [
        {
            "type": "write",
            "target": "projection-line",
            "content": "先写出原式",
        },
        {
            "type": "transform",
            "target": "projection-line",
            "content": r"\((x-2)(x-3)=0\)",
        },
        {"type": "focus", "target": "projection-line"},
        {"type": "reveal", "target": "projection-line"},
    ]


def test_compiler_adds_one_stable_path_safe_cue_to_each_fixed_beat():
    first = compile_lesson()
    second = compile_lesson()
    fixed_positions = (0, 1, -2, -1)
    expected_ids = [
        "runtime-opening-cue",
        "runtime-method-introduction-cue",
        "runtime-summary-cue",
        "runtime-transfer-intro-cue",
    ]

    for lesson in (first, second):
        assert [
            len(lesson.beats[position].sync_cues)
            for position in fixed_positions
        ] == [1, 1, 1, 1]
        assert [
            lesson.beats[position].sync_cues[0].cue_id
            for position in fixed_positions
        ] == expected_ids
        cue_ids = [
            cue.cue_id
            for beat in lesson.beats
            for cue in beat.sync_cues
        ]
        assert len(cue_ids) == len(set(cue_ids))
        assert all(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", cue_id)
            for cue_id in cue_ids
        )


def test_compiler_uses_authored_fixed_section_cues_without_rewriting_them():
    draft_payload = valid_draft()
    authored = {
        "opening_sync_cues": [
            {
                "cue_id": "script-opening",
                "spoken_text": "先看题目给出的根。",
                "lead_actions": [
                    {
                        "surface": "problem",
                        "type": "focus",
                        "target": "problem-math-001",
                    }
                ],
            }
        ],
        "method_introduction_sync_cues": [
            {
                "cue_id": "script-method",
                "spoken_text": "根的定义告诉我们可以代入。",
                "start_actions": [
                    {
                        "surface": "board",
                        "type": "write",
                        "target": "method-line",
                        "content": "代入已知根",
                    }
                ],
            }
        ],
        "summary_sync_cues": [
            {
                "cue_id": "script-summary",
                "spoken_text": "代入、整理，再回到目标。",
                "end_actions": [
                    {
                        "surface": "board",
                        "type": "fade",
                        "target": "method-line",
                    }
                ],
            }
        ],
    }
    draft_payload.update(authored)

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft_payload),
        {"review_status": "approved"},
    )

    fixed_beats = (lesson.beats[0], lesson.beats[1], lesson.beats[-2])
    expected_groups = tuple(
        [
            NarrativeSyncCue.model_validate(cue).model_dump(
                exclude_none=True
            )
            for cue in cues
        ]
        for cues in authored.values()
    )
    for beat, expected in zip(fixed_beats, expected_groups):
        assert [cue.model_dump(exclude_none=True) for cue in beat.sync_cues] == expected
        assert beat.narration == "".join(item["spoken_text"] for item in expected)


def test_compiler_preserves_authored_base_layer_for_method_section():
    draft = valid_draft()
    draft["method_introduction_sync_cues"] = [
        {
            "cue_id": "method-authored-base",
            "spoken_text": "先按主板上的关系继续推理。",
        }
    ]
    draft["fixed_section_layers_by_cue"] = {
        "method-authored-base": "base"
    }

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    method = next(beat for beat in lesson.beats if beat.purpose == "先认识方法")
    assert method.layer == "base"


def test_compiler_preserves_authored_comparison_layer_for_summary_section():
    draft = valid_draft()
    draft["summary_sync_cues"] = [
        {
            "cue_id": "summary-authored-comparison",
            "spoken_text": "在比较层上总结这两种关系。",
        }
    ]
    draft["fixed_section_layers_by_cue"] = {
        "summary-authored-comparison": "comparison"
    }

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    summary = next(beat for beat in lesson.beats if beat.purpose == "压缩方法")
    assert summary.layer == "comparison"


def test_authored_fixed_section_splits_at_interaction_and_layer_boundaries():
    draft = valid_draft()
    draft["method_introduction_sync_cues"] = [
        {"cue_id": "method-base", "spoken_text": "先留在主板判断。"},
        {
            "cue_id": "method-comparison",
            "spoken_text": "互动后进入比较层继续。",
        },
    ]
    draft["fixed_section_layers_by_cue"] = {
        "method-base": "base",
        "method-comparison": "comparison",
    }
    interaction = dict(draft["moments"][0]["interaction"])
    interaction["interaction_id"] = "fixed-method-choice"
    draft["fixed_section_interactions_after_cue"] = {
        "method-base": interaction
    }

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    method_beats = [
        beat for beat in lesson.beats if beat.purpose == "先认识方法"
    ]
    assert [beat.layer for beat in method_beats] == ["base", "comparison"]
    assert method_beats[0].interaction.interaction_id == "fixed-method-choice"
    assert method_beats[1].interaction is None
    assert [cue.cue_id for cue in method_beats[0].sync_cues] == [
        "method-base"
    ]
    assert [cue.cue_id for cue in method_beats[1].sync_cues] == [
        "method-comparison"
    ]


def test_compiler_legacy_draft_fixed_sections_remain_byte_equivalent():
    draft = LessonDraft.model_validate(valid_draft())
    lesson = LessonCompiler(lesson_id_factory=lambda: "legacy-fixed").compile(
        problem(), draft, {"review_status": "approved"}
    )
    expected = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "legacy_runtime_lesson_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert lesson.model_dump(mode="json") == expected


def test_reserved_runtime_cue_ids_have_one_shared_vocabulary():
    expected = {
        "opening": "runtime-opening-cue",
        "method_introduction": "runtime-method-introduction-cue",
        "summary": "runtime-summary-cue",
        "transfer_intro": "runtime-transfer-intro-cue",
    }

    assert schemas_module.FIXED_RUNTIME_CUE_IDS == expected
    assert compiler_module.FIXED_RUNTIME_CUE_IDS is (
        schemas_module.FIXED_RUNTIME_CUE_IDS
    )
    assert prepared_adapter.RESERVED_RUNTIME_CUE_IDS is (
        schemas_module.RESERVED_RUNTIME_CUE_IDS
    )


def test_compiler_rejects_director_cue_collision_with_reserved_runtime_id():
    draft = valid_draft()
    draft["moments"][0]["sync_cues"][0]["cue_id"] = (
        "runtime-opening-cue"
    )

    with pytest.raises(LessonCompileError, match="同步提示 ID"):
        LessonCompiler().compile(
            problem(),
            LessonDraft.model_validate(draft),
            {"review_status": "approved"},
        )


def test_compiler_appends_summary_then_transfer_interaction():
    draft = valid_draft()
    lesson = compile_lesson()
    summary = lesson.beats[-2]
    transfer = lesson.beats[-1]

    assert summary.purpose == "压缩方法"
    assert summary.board_actions[0].type == "write"
    assert summary.board_actions[0].target == "method_summary"
    assert summary.board_actions[0].content == draft["summary"]
    assert transfer.purpose == "完成近迁移"
    assert transfer.layer == "interaction"
    assert transfer.interaction.kind == "choice"
    assert transfer.interaction.prompt == draft["transfer_item"]["problem_text"]
    assert (
        transfer.interaction.expected_answer
        == draft["transfer_item"]["correct_option_id"]
    )
    assert [
        option.model_dump(exclude_none=True)
        for option in transfer.interaction.options
    ] == [
        {
            key: option[key]
            for key in ("option_id", "label", "feedback")
        }
        for option in draft["transfer_item"]["options"]
    ]
    assert transfer.interaction.hints == [
        draft["transfer_item"]["method_signal"]
    ]
    assert (
        transfer.interaction.explanation_after_correct
        == "你已经识别并使用了同一方法结构。"
    )


def test_compiler_emits_grounded_transfer_as_normal_choice():
    draft = valid_draft()
    draft["math_steps"] = []
    draft["teaching_route"] = {
        "verification_mode": "model_cross_checked",
        "teaching_route_fingerprint": "grounded-test",
    }
    draft["transfer_item"] = {
        "problem_text": (
            "若a（a≠0）是方程x^2-px+a=0的根，"
            "把x=a代入后首先得到哪个等式？"
        ),
        "expected_answer": "option-substitute",
        "method_signal": "把已知根代回原方程",
        "options": [
            {
                "option_id": "option-substitute",
                "label": r"\(a^2-pa+a=0\)",
                "canonical_answer": "a^2-p*a+a=0",
                "feedback": "对，根代入原方程后等式成立。",
            },
            {
                "option_id": "option-miss-square",
                "label": r"\(a-pa+a=0\)",
                "canonical_answer": "a-p*a+a=0",
                "feedback": "代入后x平方应变成a平方。",
            },
            {
                "option_id": "option-wrong-target",
                "label": r"\(x^2-pa+a=0\)",
                "canonical_answer": "x^2-p*a+a=0",
                "feedback": "这里还没有把x替换为已知根a。",
            },
        ],
        "correct_option_id": "option-substitute",
    }

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"verification_mode": "model_cross_checked"},
    )

    interaction = lesson.beats[-1].interaction
    assert interaction.kind == "choice"
    assert interaction.expected_answer == "option-substitute"
    assert [option.label for option in interaction.options] == [
        r"\(a^2-pa+a=0\)",
        r"\(a-pa+a=0\)",
        r"\(x^2-pa+a=0\)",
    ]


def test_compiler_preserves_legacy_text_transfer_without_options():
    draft = valid_draft()
    draft["transfer_item"]["options"] = []
    draft["transfer_item"]["correct_option_id"] = None
    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    transfer = lesson.beats[-1].interaction
    assert transfer.kind == "transfer"
    assert transfer.expected_answer == draft["transfer_item"]["expected_answer"]
    assert transfer.options == []


def test_compiler_rejects_missing_canonicalized_transfer_label_safely():
    draft = valid_draft()
    private_option_id = "private-option-without-label"
    draft["transfer_item"]["options"][0]["option_id"] = private_option_id
    draft["transfer_item"]["correct_option_id"] = private_option_id
    draft["transfer_item"]["options"][0].pop("label")

    with pytest.raises(LessonCompileError) as exc_info:
        LessonCompiler().compile(
            problem(),
            LessonDraft.model_validate(draft),
            {"review_status": "approved"},
        )

    assert str(exc_info.value) == "近迁移选项缺少已规范化的显示标签。"
    assert private_option_id not in str(exc_info.value)


def test_compiler_keeps_max_budget_method_narration_within_beat_limit():
    draft = valid_draft()
    introduction = draft["method_introduction"]
    introduction["method_name"] = "方" * 8
    introduction["student_definition"] = "定义" * 18
    introduction["target_form"] = "t" * 80
    introduction["why_it_helps"] = "作用" * 16

    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    assert len(lesson.beats[1].narration) <= 90
    assert introduction["student_definition"] in lesson.beats[1].narration
    assert introduction["why_it_helps"] in lesson.beats[1].narration


@pytest.mark.parametrize(
    "unsafe_id",
    ["", "   ", "../lesson", "lesson/other", "lesson\nother", "x" * 129],
)
def test_compiler_rejects_unsafe_injected_lesson_id(unsafe_id):
    compiler = LessonCompiler(lesson_id_factory=lambda: unsafe_id)

    with pytest.raises(LessonCompileError, match="lesson id"):
        compile_lesson(compiler)


def test_compiler_hides_lesson_id_factory_errors():
    def broken_factory():
        raise RuntimeError("private-id-detail")

    compiler = LessonCompiler(lesson_id_factory=broken_factory)

    with pytest.raises(LessonCompileError) as exc_info:
        compile_lesson(compiler)

    assert "private-id-detail" not in str(exc_info.value)


def test_compiler_preserves_step_display_and_actions_verbatim():
    draft_payload = valid_draft()
    authored = {
        "cue_id": "step-aware-cue",
        "teaching_step_id": "teaching-step-1",
        "display_text": "方程的根代入后等式成立。",
        "spoken_text": "方程的根代入以后，等式成立。",
        "lead_actions": [],
        "start_actions": [
            {
                "surface": "board",
                "type": "reveal_step_header",
                "target": "teaching-step-1",
                "teaching_step_id": "teaching-step-1",
                "step_label": "第一步：理解方程的根",
            },
            {
                "surface": "board",
                "type": "write",
                "target": "board-root",
                "content": "根代入后等式成立",
                "teaching_step_id": "teaching-step-1",
                "board_role": "knowledge_anchor",
            },
        ],
        "end_actions": [
            {
                "surface": "board",
                "type": "complete_step",
                "target": "teaching-step-1",
                "teaching_step_id": "teaching-step-1",
            }
        ],
    }
    draft_payload["moments"][0]["sync_cues"] = [authored]
    lesson = LessonCompiler(lesson_id_factory=lambda: "step-aware").compile(
        problem(),
        LessonDraft.model_validate(draft_payload),
        {"review_status": "approved"},
    )
    runtime = next(
        cue
        for beat in lesson.beats
        for cue in beat.sync_cues
        if cue.cue_id == "step-aware-cue"
    )

    assert runtime.model_dump(exclude_none=True) == (
        NarrativeSyncCue.model_validate(authored).model_dump(
            exclude_none=True
        )
    )
