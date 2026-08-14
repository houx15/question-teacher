import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import create_app
from app.generation import LessonQualityError
from app.llm_client import ModelResponseError
from app.tts_client import SpeechGenerationError
from scripts.smoke_live import assert_generated_lesson_contract
from scripts import smoke_live
from app.prompts import (
    MATH_ROUTE_SYSTEM,
    REFERENCE_AUDITOR_SYSTEM,
    REFERENCE_GROUNDING_SYSTEM,
)
from app.preparation_prompts import (
    CLASSROOM_DIRECTOR_SYSTEM,
    INTERACTION_DESIGNER_SYSTEM,
    LESSON_REVIEWER_SYSTEM,
    SCRIPT_TEACHER_SYSTEM,
    SOLUTION_TRACE_SYSTEM,
    STUDENT_SIMULATOR_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    TEACHING_PROGRESSION_SYSTEM,
)


PREPARATION_SYSTEM_PROMPTS = [
    SOLUTION_TRACE_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    TEACHING_PROGRESSION_SYSTEM,
    INTERACTION_DESIGNER_SYSTEM,
    SCRIPT_TEACHER_SYSTEM,
    CLASSROOM_DIRECTOR_SYSTEM,
    STUDENT_SIMULATOR_SYSTEM,
    LESSON_REVIEWER_SYSTEM,
]


def test_smoke_recording_client_proxies_structured_model_completion():
    class StructuredDelegate:
        def __init__(self):
            self.calls = []

        async def complete_model_with_metadata(
            self, system_prompt, user_prompt, model_type
        ):
            self.calls.append((system_prompt, user_prompt, model_type))
            return {"structured": True}

        async def complete_json(self, _system_prompt, _user_prompt):
            raise AssertionError("structured delegate must not use JSON fallback")

    delegate = StructuredDelegate()
    client = smoke_live.RecordingModelClient(delegate)

    result = asyncio.run(
        client.complete_model_with_metadata("system-a", "user-a", dict)
    )

    assert result == {"structured": True}
    assert delegate.calls == [("system-a", "user-a", dict)]
    assert client.system_prompts == ["system-a"]


def test_smoke_recording_client_explicitly_falls_back_to_json_completion():
    class JsonOnlyDelegate:
        def __init__(self):
            self.calls = []

        async def complete_json(self, system_prompt, user_prompt):
            self.calls.append((system_prompt, user_prompt))
            return {"fallback": True}

    delegate = JsonOnlyDelegate()
    client = smoke_live.RecordingModelClient(delegate)

    result = asyncio.run(
        client.complete_model_with_metadata("system-b", "user-b", dict)
    )

    assert result == {"fallback": True}
    assert delegate.calls == [("system-b", "user-b")]
    assert client.system_prompts == ["system-b"]


def page_client():
    return TestClient(create_app())


def test_live_smoke_asserts_method_first_choice_contract_without_answers():
    lesson = _smoke_contract_lesson()

    summary = assert_generated_lesson_contract(lesson)

    assert summary == {
        "method_first": True,
        "interaction_kinds": ["choice", "choice"],
        "diagnostic_choice_count": 2,
        "option_feedback_audio_ready": True,
        "formula_labels_ready": True,
        "audio_ready": True,
    }
    assert "expected_answer" not in summary


def test_live_smoke_uses_cue_audio_for_cue_based_core_lessons():
    lesson = _smoke_contract_lesson()
    for index, beat in enumerate(lesson.beats):
        beat.audio_url = None
        beat.sync_cues = [
            SimpleNamespace(audio_url=f"/audio/core-cue-{index}.mp3")
        ]

    assert assert_generated_lesson_contract(lesson)["audio_ready"] is True


def test_live_smoke_still_requires_beat_audio_for_legacy_lessons():
    lesson = _smoke_contract_lesson()
    lesson.beats[0].audio_url = None

    with pytest.raises(smoke_live.SmokeContractError, match="讲解语音"):
        assert_generated_lesson_contract(lesson)


@pytest.mark.parametrize(
    ("action_type", "target"),
    [
        ("focus", "method_name"),
        ("write", "method_target_form"),
    ],
)
def test_live_smoke_rejects_method_action_without_exact_semantic_shape(
    action_type,
    target,
):
    lesson = _smoke_contract_lesson()
    lesson.beats[1].board_actions[0] = SimpleNamespace(
        type=action_type,
        target=target,
        content="配方法",
    )

    with pytest.raises(RuntimeError, match="首个板书动作"):
        assert_generated_lesson_contract(lesson)


@pytest.mark.parametrize("unsafe_latex", (r"\frac", r"\,", r"\)", r"\("))
def test_live_smoke_rejects_any_backslash_in_method_narration(
    unsafe_latex,
):
    lesson = _smoke_contract_lesson()
    lesson.beats[1].narration = f"今天用配方法，先写 {unsafe_latex}。"

    with pytest.raises(RuntimeError, match="反斜杠"):
        assert_generated_lesson_contract(lesson)


def test_live_smoke_accepts_normal_spoken_chinese_method_narration():
    lesson = _smoke_contract_lesson()
    lesson.beats[1].narration = "今天用配方法，先把式子变成完全平方的形式。"

    assert assert_generated_lesson_contract(lesson)["method_first"] is True


@pytest.mark.parametrize(
    "invalid_label",
    (
        r"\\(x=1\\)",
        r"\)x=1\(",
        r"\(x=1",
        "x=1",
        r"\(x=99\)",
    ),
)
def test_live_smoke_rejects_noncanonical_near_transfer_labels(
    invalid_label,
):
    lesson = _smoke_contract_lesson()
    lesson.beats[-1].interaction.options[0].label = invalid_label

    with pytest.raises(RuntimeError, match="近迁移选项"):
        assert_generated_lesson_contract(lesson)


def test_live_smoke_rejects_near_transfer_option_order_or_id_mismatch():
    lesson = _smoke_contract_lesson()
    lesson.beats[-1].interaction.options[0].option_id = "wrong-order"

    with pytest.raises(RuntimeError, match="近迁移选项"):
        assert_generated_lesson_contract(lesson)


def test_live_smoke_rejects_reversed_near_transfer_option_order():
    lesson = _smoke_contract_lesson()
    lesson.beats[-1].interaction.options.reverse()

    with pytest.raises(RuntimeError, match="近迁移选项"):
        assert_generated_lesson_contract(lesson)


def test_live_smoke_defaults_to_core_and_requires_explicit_audit_flag():
    assert smoke_live.parse_args([]).with_reference_audit is False
    assert smoke_live.parse_args(["--with-reference-audit"]).with_reference_audit is True
    assert smoke_live.smoke_problem(False).reference_solution_text is None
    assert smoke_live.smoke_problem(True).reference_solution_text is not None


def test_live_smoke_accepts_grounded_parameter_root_flag_and_exact_fixture():
    args = smoke_live.parse_args(["--grounded-parameter-root"])
    problem = smoke_live.smoke_problem(
        with_reference_audit=False,
        grounded_parameter_root=args.grounded_parameter_root,
    )

    assert args.grounded_parameter_root is True
    assert "2n" in problem.problem_text
    assert "n\\ne 0" in problem.problem_text
    assert problem.reference_answer == r"$\frac{1}{2}$"
    assert problem.reference_solution_text == (
        "因为 $2n(n\\ne 0)$ 是关于x的方程"
        "$x^2-2mx+2n=0$的解\n"
        "所以 $4n^2-4mn+2n=0$\n"
        "所以$4n-4m+2=0$\n"
        "所以$m-n=\\frac{1}{2}$"
    )
    assert problem.required_method is None


def _grounded_smoke_contract_lesson():
    choice = SimpleNamespace(
        kind="choice",
        options=[
            SimpleNamespace(
                option_id=f"option-{value}",
                label=value,
                feedback="诊断反馈。",
                feedback_audio_url=f"/audio/option-{value}.mp3",
            )
            for value in ("a", "b", "c")
        ],
    )
    problem_lead = SimpleNamespace(
        surface="problem",
        type="emphasize",
        target="problem-math-001",
        content=None,
        source=None,
        relation_target=None,
        annotation=None,
        emphasis_style="underline",
        persistence="trace",
    )
    substitution_start = SimpleNamespace(
        surface="board",
        type="write",
        target="substitution-result",
        content=r"$4n^2-4mn+2n=0$",
        source=None,
        relation_target=None,
        annotation=None,
        emphasis_style=None,
        persistence=None,
    )
    return SimpleNamespace(
        lesson_id="grounded-smoke-lesson",
        beats=[
            SimpleNamespace(
                interaction=None,
                board_actions=[],
                audio_url=None,
                sync_cues=[
                    SimpleNamespace(
                        cue_id="cue-opening",
                        spoken_text="先看题目条件。",
                        audio_url="/audio/cue-opening.mp3",
                        lead_actions=[problem_lead],
                        start_actions=[],
                        end_actions=[],
                    )
                ],
            ),
            SimpleNamespace(
                interaction=choice,
                board_actions=[
                    SimpleNamespace(
                        type="write",
                        content=r"$m-n=\frac{1}{2}$",
                    )
                ],
                audio_url=None,
                sync_cues=[
                    SimpleNamespace(
                        cue_id="cue-substitution",
                        spoken_text="把根代回方程。",
                        audio_url="/audio/cue-conclusion.mp3",
                        lead_actions=[],
                        start_actions=[substitution_start],
                        end_actions=[],
                    )
                ],
            ),
        ],
        transfer_item=SimpleNamespace(
            options=[
                SimpleNamespace(
                    option_id=option.option_id,
                    label=option.label,
                )
                for option in choice.options
            ]
        ),
        validation_report={
            "verification_mode": "model_cross_checked",
            "consistency_status": "consistent",
            "teaching_route_fingerprint": "a" * 64,
            "review_status": "approved",
        },
    )


def test_grounded_parameter_root_contract_checks_route_choice_audio_and_conclusion():
    lesson = _grounded_smoke_contract_lesson()

    summary = smoke_live.assert_grounded_parameter_root_contract(lesson)

    assert summary == {
        "lesson_id": "grounded-smoke-lesson",
        "beat_count": 2,
        "cue_count": 2,
        "interaction_kinds": ["choice"],
        "review_status": "approved",
        "audio_ready": True,
        "conclusion_present": True,
    }


def test_grounded_parameter_root_contract_uses_cue_audio_without_beat_audio():
    lesson = _grounded_smoke_contract_lesson()
    assert all(beat.audio_url is None for beat in lesson.beats)

    summary = smoke_live.assert_grounded_parameter_root_contract(lesson)

    assert summary["audio_ready"] is True
    assert summary["cue_count"] == 2


def test_grounded_parameter_root_contract_requires_complete_cue_sync_evidence():
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[0].sync_cues = []
    with pytest.raises(smoke_live.SmokeContractError, match="同步 Cue"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)

    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[0].sync_cues[0].audio_url = None
    with pytest.raises(smoke_live.SmokeContractError, match="Cue 语音"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)

    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[0].sync_cues[0].lead_actions = []
    with pytest.raises(smoke_live.SmokeContractError, match="题目公式"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)

    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[0].sync_cues[0].lead_actions[0].type = "fade"
    with pytest.raises(smoke_live.SmokeContractError, match="题目公式"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


@pytest.mark.parametrize(
    "equation",
    (
        r"$4n^2 - 4mn + 2n = 0$",
        r"\( 4n^2-4mn+2n=0 \)",
        r"\[4n^2 − 4mn + 2n = 0\]",
        r"$4n^{2}-4mn+2n=0$",
        "4n²−4mn+2n=0",
        r"$4n^{2}\,-4mn\;+2n\!=0$",
    ),
)
def test_grounded_parameter_root_contract_accepts_safe_substitution_variants(
    equation,
):
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].sync_cues[0].start_actions[0].content = equation

    assert smoke_live.assert_grounded_parameter_root_contract(
        lesson
    )["audio_ready"] is True


@pytest.mark.parametrize(
    "equation",
    (
        r"$14n^2-4mn+2n=0$",
        r"$4not^2-4mn+2n=0$",
        r"$4n^2-4mn+2n=1$",
        r"$4n^2+4mn+2n=0$",
        r"$5n^2-4mn+2n=0$",
        r"$2n(2n-2m+1)=0$",
        r"$4n^2-4mn+2n=0=0$",
        r"错误：$4n^2-4mn+2n=0$",
        r"所以 $4n^2-4mn+2n=0$",
        r"$4n^{\mathbf{2}}-4mn+2n=0$",
        r"${4n^2-4mn+2n=0}$",
    ),
)
def test_grounded_parameter_root_contract_rejects_substring_equations(
    equation,
):
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].sync_cues[0].start_actions[0].content = equation

    with pytest.raises(smoke_live.SmokeContractError, match="4n"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


@pytest.mark.parametrize(
    ("report_patch", "message"),
    [
        ({"verification_mode": "symbolic_verified"}, "验证模式"),
        ({"consistency_status": "contradiction"}, "一致性"),
        ({"teaching_route_fingerprint": ""}, "指纹"),
        ({"review_status": "revision_required"}, "整篇审稿"),
    ],
)
def test_grounded_parameter_root_contract_rejects_invalid_route_report(
    report_patch,
    message,
):
    lesson = _grounded_smoke_contract_lesson()
    lesson.validation_report.update(report_patch)

    with pytest.raises(smoke_live.SmokeContractError, match=message):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


def test_grounded_parameter_root_contract_requires_choice_only_audio_and_conclusion():
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].interaction.kind = "free_text"
    with pytest.raises(smoke_live.SmokeContractError, match="选择式互动"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)

    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].board_actions[0].content = "另一个结论"
    with pytest.raises(smoke_live.SmokeContractError, match="参考结论"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


def test_grounded_parameter_root_contract_rejects_empty_choice_options():
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[-1].interaction.options = []
    lesson.transfer_item.options = []

    with pytest.raises(smoke_live.SmokeContractError, match="选项数量"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


def test_grounded_parameter_root_contract_requires_final_transfer_choice():
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats.append(
        SimpleNamespace(
            interaction=None,
            board_actions=[],
            audio_url="/audio/after-transfer.mp3",
        )
    )

    with pytest.raises(smoke_live.SmokeContractError, match="近迁移"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


@pytest.mark.parametrize(
    "invalid_conclusion",
    (
        r"错误：m-n=\frac{1}{2}",
        r"m-n=\frac{1}{2}（错误）",
        r"m-n\ne\frac{1}{2}",
        "m-n!=0.5",
        r"m-n=\frac{1}{2}=0.5",
        r"n-m=-\frac{1}{2}",
        r"m-n\le\frac{1}{2}",
    ),
)
def test_grounded_parameter_root_contract_rejects_false_conclusion_matches(
    invalid_conclusion,
):
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].board_actions[0].content = invalid_conclusion

    with pytest.raises(smoke_live.SmokeContractError, match="参考结论"):
        smoke_live.assert_grounded_parameter_root_contract(lesson)


@pytest.mark.parametrize(
    ("action_type", "conclusion"),
    [
        ("write", r"m-n=\frac{1}{2}"),
        ("transform", "m - n = 1/2"),
        ("transform", "m-n=0.5"),
    ],
)
def test_grounded_parameter_root_contract_accepts_independent_equivalent_conclusion(
    monkeypatch,
    action_type,
    conclusion,
):
    lesson = _grounded_smoke_contract_lesson()
    lesson.beats[1].board_actions[0].type = action_type
    lesson.beats[1].board_actions[0].content = conclusion

    def fail_if_grounded_uses_symbolic_label(*_args, **_kwargs):
        raise AssertionError("grounded smoke called format_answer_label")

    monkeypatch.setattr(
        smoke_live.MathEngine,
        "format_answer_label",
        fail_if_grounded_uses_symbolic_label,
    )

    assert smoke_live.assert_grounded_parameter_root_contract(
        lesson
    )["conclusion_present"] is True


def test_grounded_parameter_root_cli_prints_only_safe_contract_fields(
    monkeypatch,
    capsys,
):
    model_client = _LifecycleClient()
    speech_client = _LifecycleClient()
    _configure_smoke_clients(
        monkeypatch,
        model_client,
        speech_client,
    )
    lesson = _grounded_smoke_contract_lesson()
    generated_problems = []

    class SuccessfulGenerationService:
        def __init__(self, *_args):
            pass

        async def generate(self, problem):
            generated_problems.append(problem)
            return lesson

    class SuccessfulAudioService:
        def __init__(self, *_args):
            pass

        async def attach_audio(self, candidate):
            return candidate

    monkeypatch.setattr(
        smoke_live,
        "LessonGenerationService",
        SuccessfulGenerationService,
    )
    monkeypatch.setattr(
        smoke_live,
        "LessonAudioService",
        SuccessfulAudioService,
    )
    monkeypatch.setattr(
        smoke_live,
        "assert_model_call_contract",
        lambda *_args, **_kwargs: None,
    )

    asyncio.run(smoke_live.main(["--grounded-parameter-root"]))

    output = capsys.readouterr().out
    summary = json.loads(output)
    assert set(summary) == {
        "lesson_id",
        "beat_count",
        "cue_count",
        "interaction_kinds",
        "review_status",
        "audio_ready",
        "conclusion_present",
    }
    assert generated_problems[0].problem_text not in output
    assert generated_problems[0].reference_answer not in output
    assert generated_problems[0].reference_solution_text not in output
    assert "option-a" not in output
    assert model_client.close_calls == 1
    assert speech_client.close_calls == 1


def test_reference_audit_cli_passes_mode_to_model_call_contract(
    monkeypatch,
):
    model_client = _LifecycleClient()
    speech_client = _LifecycleClient()
    _configure_smoke_clients(
        monkeypatch,
        model_client,
        speech_client,
    )
    lesson = _smoke_contract_lesson()
    lesson.lesson_id = "reference-audit-smoke"
    lesson.validation_report = {
        "reference_material_status": "approved",
        "math_status": "verified",
        "review_status": "approved",
        "revision_count": 0,
        "math_route_source": "deterministic",
    }
    contract_modes = []

    class SuccessfulGenerationService:
        def __init__(self, *_args):
            pass

        async def generate(self, _problem):
            return lesson

    class SuccessfulAudioService:
        def __init__(self, *_args):
            pass

        async def attach_audio(self, candidate):
            return candidate

    def record_contract_mode(
        _prompts,
        grounded_parameter_root=False,
        with_reference_audit=False,
    ):
        contract_modes.append(
            (grounded_parameter_root, with_reference_audit)
        )

    monkeypatch.setattr(
        smoke_live,
        "LessonGenerationService",
        SuccessfulGenerationService,
    )
    monkeypatch.setattr(
        smoke_live,
        "LessonAudioService",
        SuccessfulAudioService,
    )
    monkeypatch.setattr(
        smoke_live,
        "assert_model_call_contract",
        record_contract_mode,
    )

    asyncio.run(smoke_live.main(["--with-reference-audit"]))

    assert contract_modes == [(False, True)]


def test_live_smoke_requires_the_complete_preparation_role_chain():
    smoke_live.assert_model_call_contract(PREPARATION_SYSTEM_PROMPTS)


def test_live_smoke_rejects_interaction_and_script_in_the_wrong_order():
    swapped = list(PREPARATION_SYSTEM_PROMPTS)
    swapped[3], swapped[4] = swapped[4], swapped[3]

    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(swapped)


def test_grounded_live_smoke_runs_grounder_before_preparation_roles():
    smoke_live.assert_model_call_contract(
        [REFERENCE_GROUNDING_SYSTEM, *PREPARATION_SYSTEM_PROMPTS],
        grounded_parameter_root=True,
    )


def test_reference_audit_live_smoke_runs_auditor_before_preparation_roles():
    smoke_live.assert_model_call_contract(
        [REFERENCE_AUDITOR_SYSTEM, *PREPARATION_SYSTEM_PROMPTS],
        with_reference_audit=True,
    )


def test_live_smoke_rejects_an_unrequested_reference_audit_prefix():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [REFERENCE_AUDITOR_SYSTEM, *PREPARATION_SYSTEM_PROMPTS]
        )


def test_grounded_live_smoke_rejects_an_auditor_after_the_grounder():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                REFERENCE_GROUNDING_SYSTEM,
                REFERENCE_AUDITOR_SYSTEM,
                *PREPARATION_SYSTEM_PROMPTS,
            ],
            grounded_parameter_root=True,
            with_reference_audit=True,
        )


def test_live_smoke_rejects_a_stray_incomplete_repair_suffix():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                *PREPARATION_SYSTEM_PROMPTS,
                CLASSROOM_DIRECTOR_SYSTEM,
            ]
        )


def test_live_smoke_accepts_more_than_two_complete_targeted_repairs():
    performance_repair = PREPARATION_SYSTEM_PROMPTS[5:]

    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS,
            *performance_repair,
            *performance_repair,
            *performance_repair,
        ]
    )


def test_live_smoke_accepts_the_pipeline_repair_cycle_limit():
    performance_repair = PREPARATION_SYSTEM_PROMPTS[5:]

    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS,
            *(performance_repair * smoke_live.LessonPreparationPipeline.MAX_REPAIR_CYCLES),
        ]
    )


def test_live_smoke_rejects_more_than_the_pipeline_repair_cycle_limit():
    performance_repair = PREPARATION_SYSTEM_PROMPTS[5:]

    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                *PREPARATION_SYSTEM_PROMPTS,
                *(performance_repair * (smoke_live.LessonPreparationPipeline.MAX_REPAIR_CYCLES + 1)),
            ]
        )


def test_live_smoke_accepts_one_fresh_final_review_call():
    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS,
            LESSON_REVIEWER_SYSTEM,
        ]
    )


@pytest.mark.parametrize("review_call_count", [3, 4, 5, 6])
def test_live_smoke_accepts_bounded_review_retries_after_a_complete_repair(
    review_call_count,
):
    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS,
            *PREPARATION_SYSTEM_PROMPTS[5:7],
            *([LESSON_REVIEWER_SYSTEM] * review_call_count),
        ]
    )


def test_live_smoke_rejects_four_reviews_without_a_repair_context():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                *PREPARATION_SYSTEM_PROMPTS[:-1],
                LESSON_REVIEWER_SYSTEM,
                LESSON_REVIEWER_SYSTEM,
                LESSON_REVIEWER_SYSTEM,
                LESSON_REVIEWER_SYSTEM,
            ]
        )


def test_live_smoke_rejects_seven_reviews_after_a_complete_repair():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                *PREPARATION_SYSTEM_PROMPTS,
                *PREPARATION_SYSTEM_PROMPTS[5:7],
                *([LESSON_REVIEWER_SYSTEM] * 7),
            ]
        )


@pytest.mark.parametrize("call_count", [2, 3])
@pytest.mark.parametrize("role_index", range(8))
def test_live_smoke_accepts_bounded_structural_retries_for_each_role(
    role_index,
    call_count,
):
    role_prompt = PREPARATION_SYSTEM_PROMPTS[role_index]

    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS[:role_index],
            *([role_prompt] * call_count),
            *PREPARATION_SYSTEM_PROMPTS[role_index + 1 :],
        ]
    )


@pytest.mark.parametrize("repair_start", range(6))
def test_live_smoke_accepts_each_legal_targeted_repair_start(repair_start):
    smoke_live.assert_model_call_contract(
        [
            *PREPARATION_SYSTEM_PROMPTS,
            *PREPARATION_SYSTEM_PROMPTS[repair_start:],
        ]
    )


def test_live_smoke_rejects_a_fourth_structural_attempt():
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(
            [
                PREPARATION_SYSTEM_PROMPTS[0],
                PREPARATION_SYSTEM_PROMPTS[0],
                PREPARATION_SYSTEM_PROMPTS[0],
                *PREPARATION_SYSTEM_PROMPTS,
            ]
        )


@pytest.mark.parametrize(
    "calls",
    [
        PREPARATION_SYSTEM_PROMPTS[:-1],
        [
            PREPARATION_SYSTEM_PROMPTS[1],
            PREPARATION_SYSTEM_PROMPTS[0],
            *PREPARATION_SYSTEM_PROMPTS[2:],
        ],
        [*PREPARATION_SYSTEM_PROMPTS, "obsolete-whole-lesson-agent"],
    ],
)
def test_live_smoke_rejects_incomplete_out_of_order_or_unknown_roles(calls):
    with pytest.raises(smoke_live.SmokeContractError):
        smoke_live.assert_model_call_contract(calls)


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            LessonQualityError("private lesson output and prompt"),
            "讲解生成未通过质量门，请检查模型输出后重试。",
        ),
        (
            ModelResponseError("private provider body"),
            "模型服务调用失败，请检查配置或稍后重试。",
        ),
        (
            SpeechGenerationError("private tts response"),
            "语音生成失败，请检查 TTS 配置或稍后重试。",
        ),
        (
            httpx.ConnectError("private network endpoint"),
            "现场服务网络请求失败，请检查网络连接或稍后重试。",
        ),
    ],
)
def test_live_smoke_cli_converts_known_failures_to_safe_errors(
    monkeypatch,
    error,
    expected_message,
):
    async def fail_safely(_argv=None):
        raise error

    monkeypatch.setattr(smoke_live, "main", fail_safely)

    with pytest.raises(SystemExit) as exc_info:
        smoke_live.run_cli([])

    assert str(exc_info.value) == expected_message
    assert "private" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True


class _LifecycleClient:
    def __init__(self, close_error=None):
        self.close_error = close_error
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _configure_smoke_clients(monkeypatch, model_client, speech_client):
    monkeypatch.setattr(
        smoke_live.Settings,
        "from_env",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(smoke_live, "missing_environment", lambda _: [])
    monkeypatch.setattr(
        smoke_live,
        "OpenAICompatibleClient",
        lambda _settings: model_client,
    )
    monkeypatch.setattr(
        smoke_live,
        "create_speech_client",
        lambda _settings: speech_client,
    )


def test_live_smoke_real_generation_service_reaches_model_cli_category(
    monkeypatch,
):
    class ProviderFailingClient(_LifecycleClient):
        def __init__(self):
            super().__init__()
            self.complete_calls = 0

        async def complete_json(self, _system_prompt, _user_prompt):
            self.complete_calls += 1
            raise ModelResponseError("private provider response")

    model_client = ProviderFailingClient()
    speech_client = _LifecycleClient()
    _configure_smoke_clients(
        monkeypatch,
        model_client,
        speech_client,
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke_live.run_cli([])

    assert str(exc_info.value) == "模型服务调用失败，请检查配置或稍后重试。"
    assert "private" not in str(exc_info.value)
    assert model_client.complete_calls >= 1
    assert model_client.close_calls == 1
    assert speech_client.close_calls == 1


def test_live_smoke_closes_model_when_speech_client_construction_fails(
    monkeypatch,
):
    model_client = _LifecycleClient()
    monkeypatch.setattr(
        smoke_live.Settings,
        "from_env",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(smoke_live, "missing_environment", lambda _: [])
    monkeypatch.setattr(
        smoke_live,
        "OpenAICompatibleClient",
        lambda _settings: model_client,
    )

    def fail_speech_construction(_settings):
        raise SpeechGenerationError("private construction detail")

    monkeypatch.setattr(
        smoke_live,
        "create_speech_client",
        fail_speech_construction,
    )

    with pytest.raises(SpeechGenerationError):
        asyncio.run(smoke_live.main([]))

    assert model_client.close_calls == 1


def test_live_smoke_body_failure_wins_while_both_clients_close_once(
    monkeypatch,
):
    body_error = LessonQualityError("primary quality failure")
    model_client = _LifecycleClient(
        ModelResponseError("private model close detail")
    )
    speech_client = _LifecycleClient(
        SpeechGenerationError("private speech close detail")
    )
    _configure_smoke_clients(
        monkeypatch,
        model_client,
        speech_client,
    )

    class FailingGenerationService:
        def __init__(self, *_args):
            pass

        async def generate(self, _problem):
            raise body_error

    monkeypatch.setattr(
        smoke_live,
        "LessonGenerationService",
        FailingGenerationService,
    )

    with pytest.raises(LessonQualityError) as exc_info:
        asyncio.run(smoke_live.main([]))

    assert exc_info.value is body_error
    assert model_client.close_calls == 1
    assert speech_client.close_calls == 1


def test_live_smoke_close_only_failure_is_safely_classified(
    monkeypatch,
    capsys,
):
    model_client = _LifecycleClient(
        httpx.ConnectError("private close endpoint")
    )
    speech_client = _LifecycleClient()
    _configure_smoke_clients(
        monkeypatch,
        model_client,
        speech_client,
    )
    lesson = SimpleNamespace(
        lesson_id="smoke-lesson",
        beats=[],
        validation_report={},
    )

    class SuccessfulGenerationService:
        def __init__(self, *_args):
            pass

        async def generate(self, _problem):
            return lesson

    class SuccessfulAudioService:
        def __init__(self, *_args):
            pass

        async def attach_audio(self, candidate):
            return candidate

    monkeypatch.setattr(
        smoke_live,
        "LessonGenerationService",
        SuccessfulGenerationService,
    )
    monkeypatch.setattr(
        smoke_live,
        "LessonAudioService",
        SuccessfulAudioService,
    )
    monkeypatch.setattr(
        smoke_live,
        "assert_generated_lesson_contract",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        smoke_live,
        "assert_model_call_contract",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke_live.run_cli([])

    assert str(exc_info.value) == (
        "现场服务网络请求失败，请检查网络连接或稍后重试。"
    )
    assert "private" not in str(exc_info.value)
    assert capsys.readouterr().out == ""
    assert model_client.close_calls == 1
    assert speech_client.close_calls == 1


def test_live_smoke_uses_automatic_temporary_audio_directory():
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "smoke_live.py"
    ).read_text()

    assert "tempfile.TemporaryDirectory" in source
    assert 'REPOSITORY_ROOT / "var" / "audio"' not in source


def _smoke_contract_lesson():
    choice = SimpleNamespace(
        kind="choice",
        options=[
            SimpleNamespace(
                option_id=f"option-{value}",
                label=rf"\(x={value}\)",
                feedback="诊断反馈。",
                feedback_audio_url=f"/audio/option-{value}.mp3",
            )
            for value in ("1", "2", "3")
        ],
    )
    transfer_options = [
        SimpleNamespace(
            option_id=f"option-{value}",
            label=rf"\(x={value}\)",
            canonical_answer=f"x={value}",
        )
        for value in ("1", "2", "3")
    ]
    return SimpleNamespace(
        beats=[
            SimpleNamespace(
                purpose="进入问题",
                layer="base",
                narration="先看题目。",
                board_actions=[],
                interaction=None,
                audio_url="/audio/opening.mp3",
            ),
            SimpleNamespace(
                purpose="先认识方法",
                layer="micro_explanation",
                narration="今天用配方法。",
                board_actions=[
                    SimpleNamespace(
                        type="write",
                        target="method_name",
                        content="配方法",
                    )
                ],
                interaction=None,
                audio_url="/audio/method.mp3",
            ),
            SimpleNamespace(
                purpose="诊断",
                layer="interaction",
                narration="请选择。",
                board_actions=[],
                interaction=choice,
                audio_url="/audio/diagnostic.mp3",
            ),
            SimpleNamespace(
                purpose="完成近迁移",
                layer="interaction",
                narration="现在迁移。",
                board_actions=[],
                interaction=choice,
                audio_url="/audio/transfer.mp3",
            ),
        ],
        transfer_item=SimpleNamespace(options=transfer_options),
    )


def test_readme_documents_method_first_choice_generation_and_local_katex():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    support = readme.split("## 运行环境", maxsplit=1)[0]
    vendor = readme.split("## 本地公式资源", maxsplit=1)[1].split(
        "## 自动化验证", maxsplit=1
    )[0]
    validation = readme.split("## 自动化验证", maxsplit=1)[1].split(
        "## 证据边界", maxsplit=1
    )[0]

    assert "先认识方法" in support
    assert "配方法" in support
    assert "choice" in support and "point_select" in support
    assert "free_text" in support and "needs_review" in support
    assert "Director" in support and "Reviewer" in support
    assert "确定性门禁" in support and "prompt 的语义判断" in support
    assert "每个选择项都有针对该选项的诊断反馈" in support

    assert "CDN" in vendor and "package-lock.json" in vendor
    for command in (
        "npm install",
        "cp node_modules/katex/dist/katex.mjs",
        "cp node_modules/katex/dist/katex.min.css",
        "cp -R node_modules/katex/dist/fonts",
        "cp node_modules/katex/LICENSE",
        "npm test",
    ):
        assert command in vendor

    assert "pytest -q tests" in validation
    assert "python -m compileall -q app scripts tests" in validation
    assert "npm test" in validation
    assert "node --check app/static/lesson.js" in validation
    assert "python scripts/smoke_live.py" in validation
    assert "python scripts/smoke_live.py --with-reference-audit" in validation
    assert "临时目录" in validation and "var/audio" in validation
    assert "选项诊断反馈语音" in validation


def test_readme_documents_reference_grounded_scope_and_parameter_example():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    for boundary in (
        "参考材料依据",
            "不是自动批改系统",
            "严格符号校验",
            "结构化模型审阅",
            "严格符号路径重现了数学矛盾",
            "局部检查 `failed`",
            "仅保留在服务端",
    ):
        assert boundary in readme

    for input_field in (
        "problem_text",
        "reference_answer",
        "reference_solution_text",
    ):
        assert input_field in readme

    assert "2n" in readme and "m-n" in readme
    assert "python scripts/smoke_live.py --grounded-parameter-root" in readme
    assert "参数已实现" in readme
    assert "本轮真实执行" in readme
    assert "当前版本不可执行" not in readme


def test_readme_documents_saved_lesson_id_lifecycle():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    for persistence_contract in (
        "课程完成后",
        "SQLite 持久化",
        "`var/lessons.sqlite3`",
        "`var/audio/{lesson_id}/`",
        "生成任务仍只保存在内存",
        "课程 ID 只会在持久化保存成功后",
        "生成完成页",
        "复制课程 ID",
        "不会自动进入课堂",
        "已有课程 ID",
        "服务重启后",
        "每次都会分配新的课程 ID",
        "不按题目内容复用",
        "账号、课程列表、删除和搜索",
    ):
        assert persistence_contract in readme


def test_readme_documents_saved_lesson_backup_contract():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    for backup_contract in (
        "删除数据库中的课程记录",
        "删除对应的音频目录",
        "停止服务",
        "同时备份",
        "SQLite sidecar",
        "不要单独复制",
    ):
        assert backup_contract in readme


def test_readme_documents_cue_sync_contract_and_evidence_boundaries():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()

    assert "Cue 级" in readme and "火山引擎" in readme
    assert "公式渲染" in readme and "强调" in readme
    assert "相互独立的白名单合同" in readme
    assert "固定参数根 smoke" in readme
    assert "所有题型" in readme and "学习效果" in readme
    assert "可读降级" in readme
    assert "同步语音成功" in readme
    assert (
        "它只输出课程 ID、Beat/Cue 数量、互动类型、审稿状态、音频就绪和"
        in readme
    )
    assert "配置的 TTS provider" in readme
    assert "火山引擎是当前默认路径" in readme
    assert "每个 Cue\n使用一段火山引擎语音" not in readme
    assert "所有 Beat 都有音频" not in readme


def test_generation_page_has_focused_authoring_form():
    response = page_client().get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    html = response.text
    assert 'id="lesson-form"' in html
    assert 'name="problem_text"' in html
    assert 'name="reference_answer"' in html
    assert 'name="reference_solution_text"' in html
    assert 'id="reference_solution_text"' in html
    assert html.count('maxlength="12000"') == 3
    assert "参考解析" in html
    assert "输入题目、参考答案与可选参考解析" in html
    assert 'name="required_method"' in html
    assert 'name="lesson_length"' in html
    assert 'id="model-status"' in html
    assert 'id="voice-status"' in html
    assert 'id="generation-progress"' in html
    assert 'src="/static/generate.js?v=20260814-1"' in html
    assert "OPENAI_API_KEY" not in html
    assert "validation_report" not in html


def test_generation_page_can_reopen_and_confirm_a_saved_lesson():
    html = page_client().get("/").text

    for element_id in (
        "existing-lesson-form",
        "existing-lesson-id",
        "existing-lesson-error",
        "generation-complete",
        "completed-lesson-id",
        "copy-lesson-id",
        "copy-lesson-status",
        "enter-completed-lesson",
        "create-another-lesson",
    ):
        assert f'id="{element_id}"' in html

    assert 'maxlength="128"' in html
    assert 'pattern="[A-Za-z0-9][A-Za-z0-9._:-]{0,127}"' in html
    assert "课程已保存" in html


def test_versioned_generation_module_remains_cacheable():
    client = page_client()
    generate_response = client.get(
        "/static/generate.js?v=20260814-1",
    )
    source = generate_response.text
    assert (
        'from "./generation-flow.mjs?v=20260810-1"'
        in source
    )

    flow_response = client.get(
        "/static/generation-flow.mjs?v=20260810-1",
    )
    for response in (generate_response, flow_response):
        assert response.status_code == 200
        cache_directives = {
            directive.partition("=")[0].strip().lower()
            for directive in response.headers.get(
                "cache-control",
                "",
            ).split(",")
            if directive.strip()
        }
        assert cache_directives.isdisjoint({"no-cache", "no-store"})


def test_generation_page_submits_optional_reference_solution():
    source = page_client().get("/static/generate.js").text
    html = page_client().get("/").text

    assert 'data.get("reference_solution_text")' in source
    assert "reference_solution_text: referenceSolution || null" in source
    public_stages = (
        "正在理解题目",
        "正在核对题目材料",
        "正在整理参考解析",
        "正在设计解题思维轨迹",
        "正在设计课堂推进",
        "正在设计互动",
        "正在编写讲稿",
        "正在编排板书与高亮",
        "正在审核和优化课程",
        "正在编译课程",
        "正在生成语音",
        "正在保存课程",
    )
    for stage in public_stages:
        assert f'"{stage}"' in source
        assert f'data-stage="{stage}"' in html
    assert (
        '"正在设计课堂推进": '
        '"把思路组织成学生能够一步步跟上的课堂结构。"'
        in source
    )
    for private_status in (
        "正在验证数学路线",
        "正在审阅参考解析",
        "verification_mode",
        "model_disagreement",
        "check_requests",
    ):
        assert private_status not in source
        assert private_status not in html


def test_lesson_page_has_fullscreen_classroom_regions():
    response = page_client().get("/lesson/example")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    html = response.text
    for region_id in (
        "classroom-shell",
        "lesson-topbar",
        "problem-display",
        "route-strip",
        "board-stage",
        "structured-board",
        "legacy-board",
        "base-board",
        "layer-stage",
        "narration-line",
        "interaction-stage",
        "lesson-controls",
        "start-overlay",
        "loading-state",
        "error-state",
        "empty-state",
        "rotate-state",
    ):
        assert f'id="{region_id}"' in html
    assert 'type="module"' in html
    assert 'src="/static/lesson.js?v=20260814-3"' in html
    assert 'href="/static/styles.css?v=20260814-1"' in html
    assert '<link rel="stylesheet" href="/static/vendor/katex/katex.min.css">' in html
    assert 'class="sidebar"' not in html


def test_versioned_lesson_module_remains_cacheable():
    client = page_client()
    html_response = client.get("/lesson/example")
    response = client.get("/static/lesson.js?v=20260814-3")
    runtime_response = client.get(
        "/static/runtime-core.mjs?v=20260814-2"
    )
    board_response = client.get(
        "/static/structured-board.mjs?v=20260814-1"
    )
    styles_response = client.get("/static/styles.css?v=20260814-1")

    assert html_response.headers["cache-control"] == "no-cache"
    assert response.status_code == 200
    assert (
        '} from "./runtime-core.mjs?v=20260814-2";'
        in response.text
    )
    assert (
        '} from "./structured-board.mjs?v=20260814-1";'
        in runtime_response.text
    )
    for asset_response in (
        response,
        runtime_response,
        board_response,
        styles_response,
    ):
        assert asset_response.status_code == 200
        cache_directives = {
            directive.partition("=")[0].strip().lower()
            for directive in asset_response.headers.get(
                "cache-control",
                "",
            ).split(",")
            if directive.strip()
        }
        assert cache_directives.isdisjoint({"no-cache", "no-store"})


def test_lesson_runtime_renders_math_and_tracks_unrendered_board_sources():
    source = page_client().get("/static/lesson.js").text

    assert "mathTextToPlainText" in source
    assert "renderMathText" in source
    assert "renderProblemMathText" in source
    assert "renderMathText(dom.title, lesson.title)" in source
    assert "function renderProblemFocus()" in source
    assert "renderProblemMathText(dom.problem, lesson.problem.problem_text" in source
    assert "focusTargets: lesson.problem_focus_targets || []" in source
    assert "visualState: visualState.problem" in source
    assert "emptyVisualState(problemTargetIds)" in source
    assert "renderMathText(dom.problem, lesson.problem.problem_text)" not in source
    assert "renderMathText(dom.narration, beat.narration)" in source
    assert "renderMathText(heading, interaction.prompt)" in source
    assert "renderMathText(button, option.label)" in source
    assert "renderMathText(ui.feedback, presentation.message)" in source
    assert "content.dataset.source" in source
    assert "content.textContent !== value.content" not in source
    assert "renderMathText(content, source)" in source


def test_lesson_runtime_uses_cue_timeline_and_preserves_legacy_playback():
    source = page_client().get("/static/lesson.js").text

    assert '} from "./runtime-core.mjs?v=20260814-2";' in source
    assert 'import { CuePlayer } from "./cue-player.mjs?v=20260814-1";' in source
    assert "applySyncVisualAction" in source
    assert "let cuePlayer = null;" in source
    assert "cuePlayer = new CuePlayer({" in source
    assert "Array.isArray(beat.sync_cues) && beat.sync_cues.length > 0" in source
    assert "cuePlayer.playBeat(beat, { snapshot })" in source
    assert "beginLegacyBeatPlayback(beat, token)" in source
    assert "function scheduleLegacyActions(" in source
    assert "scheduleBoardActions(" in source
    assert "function applyCueActions(actions)" in source
    assert "applySyncVisualAction(nextState, action)" in source
    assert "renderProblemFocus();" in source
    assert "renderActiveBoards();" in source
    assert "baseBoard: cloneBoard(runtime.baseBoard)" in source
    assert "problem: cloneBoard(visualState.problem)" in source
    assert "onBeatComplete: () => finishBeat(beatToken)" in source
    assert "onAudioUnavailable:" in source
    assert "cuePlayer?.stop();" in source
    assert "cuePlayer.pause();" in source
    assert "cuePlayer.resume();" in source
    cue_branch = source[
        source.index("function beginCueBeatPlayback("):
        source.index("function finishBeat(")
    ]
    assert "scheduleBoardActions(" not in cue_branch
    assert "setTimeout(" not in cue_branch
    assert "showInteraction(" not in cue_branch
    finish_branch = source[
        source.index("function finishBeat("):
        source.index("function leaveTemporaryLayer(")
    ]
    assert "showInteraction(beat.interaction)" in finish_branch


def test_lesson_runtime_renders_and_scrolls_the_continuous_structured_board():
    source = page_client().get("/static/lesson.js").text

    assert 'section.className = "lesson-step"' in source
    assert 'header.className = "lesson-step__header"' in source
    assert 'bullet.className = "lesson-step__status-dot"' in source
    assert 'statusText.className = "lesson-step__status-text sr-only"' in source
    assert 'section.setAttribute("aria-current", "step")' in source
    assert 'record.support.className = "lesson-step__support"' in source
    assert "syncStructuredLines(record.main, step.lines, record.lines)" in source
    assert "dom.structuredBoard.hidden = !hasStructuredBoard" in source
    assert "dom.legacyBoard.hidden = hasStructuredBoard" in source
    assert "renderStructuredBoard(structuredBoard, dom.structuredBoard)" in source
    assert "requestedScrollStepId: null" in source
    assert 'const behavior = restoringSnapshot ? "auto" : "smooth"' in source
    assert 'requested.scrollIntoView({ behavior, block: "center" })' in source
    assert "cue.display_text || cue.spoken_text" in page_client().get(
        "/static/cue-player.mjs"
    ).text
    assert "cuePlayer.playCueSequence(" in source
    assert "runSupportCueSequence(" not in source


def test_lesson_shell_uses_single_continuous_structured_board():
    client = page_client()
    html = client.get("/lesson/example").text
    css = client.get("/static/styles.css?v=20260814-1").text

    assert 'id="structured-board"' in html
    assert 'tabindex="0"' in html
    assert 'id="legacy-board"' in html
    assert 'class="board-directory"' not in html
    assert ".structured-board {" in css
    assert ".lesson-step {" in css
    assert ".lesson-step__header {" in css
    assert ".lesson-step__status-dot {" in css
    assert ".lesson-step__line {" in css
    assert ".lesson-step__support {" in css
    assert ".structured-board:focus-visible {" in css
    assert "font-size: 16px;" in css
    assert "font-size: 22px;" in css
    assert "overflow-x: hidden;" in css
    assert "overflow-y: auto;" in css
    assert "scroll-margin-block: 28vh;" in css
    assert "env(safe-area-inset-bottom)" in css
    assert "-webkit-overflow-scrolling: touch;" in css


def test_structured_board_module_cache_chain_is_versioned():
    client = page_client()
    html = client.get("/lesson/example").text
    lesson_js = client.get("/static/lesson.js?v=20260814-3").text
    runtime_js = client.get("/static/runtime-core.mjs?v=20260814-2").text

    assert "lesson.js?v=20260814-3" in html
    assert "styles.css?v=20260814-1" in html
    assert "runtime-core.mjs?v=20260814-2" in lesson_js
    assert "structured-board.mjs?v=20260814-1" in runtime_js


def test_choice_submission_passes_selected_option_without_exposing_answer_key():
    source = page_client().get("/static/lesson.js").text

    assert "submitInteraction(interaction, option.option_id, option," in source
    assert "async function submitInteraction(interaction, answer, selectedOption, ui)" in source
    assert "resolveInteractionPresentation" in source
    assert "if (result?.feedback)" in source
    assert "if (selectedOption?.feedback)" not in source
    assert "expected: interaction.expected_answer" not in source
    assert "interaction.expected_answer" not in source


def test_choice_buttons_use_nonempty_plain_text_accessible_names():
    source = page_client().get("/static/lesson.js").text

    assert "mathTextToPlainText" in source
    assert '"./math-text.mjs?v=20260807-2";' in source
    assert "for (const [optionIndex, option] of" in source
    assert "const accessibleLabel = mathTextToPlainText(option.label);" in source
    assert 'button.setAttribute(' in source
    assert '"aria-label",' in source
    assert "accessibleLabel || `选项 ${optionIndex + 1}`" in source
    assert "option.canonical_answer" not in source


def test_synchronized_emphasis_uses_fixed_classes_and_no_inner_html():
    client = page_client()
    lesson_source = client.get("/static/lesson.js").text
    math_source = client.get("/static/math-text.mjs").text
    runtime_source = client.get("/static/runtime-core.mjs").text
    styles = client.get("/static/styles.css").text

    assert ".innerHTML" not in lesson_source
    assert ".innerHTML" not in math_source
    assert (
        'import { emphasisClassName } '
        'from "./runtime-core.mjs?v=20260807-2";'
        in math_source
    )
    assert "emphasisClassName(value.emphasis?.style)" in lesson_source
    assert 'node.classList.remove(...EMPHASIS_CLASSES)' in lesson_source
    assert 'highlight: "is-highlighted"' in runtime_source
    assert 'underline: "is-underlined"' in runtime_source
    assert 'red: "is-red-emphasis"' in runtime_source
    assert "return EMPHASIS_CLASSES[style] ||" in runtime_source
    for token in (
        ".focus-target",
        ".is-highlighted.is-active",
        ".is-highlighted.is-trace",
        ".is-underlined.is-active",
        ".is-underlined.is-trace",
        ".is-red-emphasis.is-active",
        ".is-red-emphasis.is-trace",
    ):
        assert token in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles


def test_static_pages_include_accessibility_and_responsive_contracts():
    client = page_client()
    index_html = client.get("/").text
    lesson_html = client.get("/lesson/example").text
    styles = client.get("/static/styles.css")

    assert 'aria-live="polite"' in index_html
    assert 'aria-live="polite"' in lesson_html
    assert 'aria-label="暂停或继续"' in lesson_html
    assert 'aria-label="进入全屏"' in lesson_html
    assert styles.status_code == 200
    assert "@media (orientation: portrait)" in styles.text
    assert "--board:" in styles.text
    assert "--focus:" in styles.text
    assert '<meta name="theme-color" content="#f4efe5">' in lesson_html
    assert "--classroom-canvas: #f4efe5;" in styles.text
    assert "--board-surface: #fbfaf6;" in styles.text
    assert "--board-ink: #203047;" in styles.text
    assert "--classroom-panel: #fffdf8;" in styles.text


def test_interaction_submission_uses_server_authoritative_contract():
    source = page_client().get("/static/lesson.js").text

    assert "lesson_id: lesson.lesson_id" in source
    assert "interaction_id: interaction.interaction_id" in source
    assert "expected: interaction.expected_answer" not in source


def test_point_select_prompt_does_not_block_board_pointer_or_keyboard_access():
    client = page_client()
    source = client.get("/static/lesson.js").text
    styles = client.get("/static/styles.css").text

    assert 'classList.toggle("is-point-select"' in source
    assert ".interaction-stage.is-point-select" in styles
    assert "pointer-events: none" in styles
    assert ".interaction-stage.is-point-select .interaction-card" in styles
    assert "pointer-events: auto" in styles
    assert 'node.setAttribute("role", "button")' in source
    assert "const activeRegion = !dom.structuredBoard.hidden" in source
    assert "? dom.structuredBoard" in source
    assert ": runtime.layerStack.length > 0" in source
    assert 'activeRegion.querySelectorAll(".board-object")' in source
    assert "node.getClientRects().length > 0" in source
    assert 'selectablePrefix.className = "sr-only board-selectable-prefix"' in source
    assert 'selectablePrefix.textContent = "选择板书："' in source
    assert 'selectablePrefix.setAttribute("aria-hidden", "true")' in source
    assert 'prefix?.removeAttribute("aria-hidden")' in source
    assert 'prefix?.setAttribute("aria-hidden", "true")' in source
    assert 'node.setAttribute("aria-label"' not in source
    assert "humanizeTarget(node.dataset.boardTarget)" not in source
    assert "boardSource" not in source
    assert 'event.key === "Enter" || event.key === " "' in source


def test_primary_control_uses_runtime_intent_for_ended_pause_beats():
    source = page_client().get("/static/lesson.js").text

    assert "runtime.primaryControlIntent(paused)" in source
    assert 'if (intent === "advance")' in source
    assert 'if (action.type === "pause") setPaused(true)' not in source


def test_local_katex_assets_are_served_with_the_math_text_module():
    client = page_client()

    for path, media_type in (
        ("/static/math-text.mjs", "text/javascript"),
        ("/static/vendor/katex/katex.mjs", "text/javascript"),
        ("/static/vendor/katex/katex.min.css", "text/css"),
        ("/static/vendor/katex/fonts/KaTeX_Main-Regular.woff2", "font/woff2"),
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)


def test_interaction_evaluation_and_feedback_audio_have_bounded_lifecycles():
    source = page_client().get("/static/lesson.js").text

    assert "const EVALUATION_TIMEOUT_MS = 14000;" in source
    assert "const FEEDBACK_AUDIO_TIMEOUT_MS = 12000;" in source
    assert "let activeEvaluationController = null;" in source
    assert "let interactionSubmitting = false;" in source
    assert "const controller = new AbortController();" in source
    assert "activeEvaluationController = controller;" in source
    assert "activeEvaluationController === controller" in source
    assert "signal: controller.signal" in source
    assert "controller.abort()" in source
    assert "clearTimeout(timeout)" in source
    assert "createBoundedSettlement" in source
    assert "feedbackAudioFinalizer?.()" in source
    assert 'audio.addEventListener("ended", onEnded' in source
    assert 'audio.addEventListener("error", onError' in source
    assert "playAttempt?.catch(settle)" in source
    assert "window.setTimeout(resolve, 650)" not in source
    assert 'audio.removeAttribute("src")' in source
    assert "audio.load()" in source


def test_submission_state_guards_every_async_boundary_and_navigation():
    source = page_client().get("/static/lesson.js").text
    submit_source = source[
        source.index("async function submitInteraction"):
        source.index("async function toggleFullscreen")
    ]

    assert "let submissionSequence = 0;" in source
    assert "submissionSequence += 1;" in source
    assert "interactionSubmitting = true;" in submit_source
    assert "const originatingBeatToken = beatToken;" in submit_source
    assert "const originatingInteractionId = interaction.interaction_id;" in submit_source
    assert "const isCurrentSubmission = () =>" in submit_source
    assert submit_source.count("if (!isCurrentSubmission()) return;") >= 3
    assert submit_source.index("if (!isCurrentSubmission()) return;") < (
        submit_source.index("runtime.recordAnswer")
    )
    assert "interactionSubmitting = false;" in submit_source
    assert "activeEvaluationController?.abort()" in source
    assert "|| interactionSubmitting" in source


def test_global_shortcuts_preserve_native_interactive_keyboard_behavior():
    source = page_client().get("/static/lesson.js").text

    assert "isNativeInteractiveTarget(event.target)" in source
    assert "isNativeInteractiveTarget(document.activeElement)" in source
    assert 'if (!dom.previous.disabled) previousBeat();' in source
    assert 'if (!dom.replay.disabled) replayCurrentBeat();' in source


def test_needs_review_waits_for_explicit_continue_and_errors_clear_stale_hints():
    source = page_client().get("/static/lesson.js").text
    submit_source = source[
        source.index("async function submitInteraction"):
        source.index("async function toggleFullscreen")
    ]
    catch_source = submit_source.rsplit("} catch {", maxsplit=1)[1]

    assert 'presentation.advanceMode === "manual"' in submit_source
    assert "ui.continueButton.focus()" in submit_source
    assert "return;" in submit_source
    assert 'renderMathText(ui.hint, "");' in catch_source
    assert 'interaction.kind === "point_select"' in catch_source
    assert "enablePointSelection((retryAnswer)" in catch_source


def test_interaction_card_scrolls_safely_in_short_landscape():
    styles = page_client().get("/static/styles.css").text

    assert ".interaction-card {" in styles
    assert "max-height: 100%;" in styles
    assert "overflow-y: auto;" in styles
    assert ".interaction-card h2 {" in styles
    assert "overflow-x: auto;" in styles
    assert "@media (max-height: 600px) and (orientation: landscape)" in styles
    assert ".interaction-stage {" in styles
