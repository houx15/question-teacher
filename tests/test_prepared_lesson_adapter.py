import importlib.util

import pytest

from app.compiler import LessonCompiler
from app.preparation_models import PreparedLesson
from app.prepared_lesson_adapter import prepared_lesson_to_draft
from app.schemas import MathStep, ProblemInput, RuntimeLesson
from app.teaching_route import (
    TeachingRouteConsistency,
    TeachingRouteMode,
    _freeze_route,
)
from tests.test_preparation_validation import (
    overlay_score_payload,
    prepared_payload,
    route,
)


def test_prepared_lesson_adapter_module_exists():
    assert importlib.util.find_spec("app.prepared_lesson_adapter") is not None


def source_problem():
    return ProblemInput(
        problem_text="若2n（n不等于0）是方程x^2-2mx+2n=0的根，求m-n。",
        reference_answer="m-n=1/2",
        reference_solution_text="把已知根代入，利用n不等于0整理。",
    )


def approved_prepared():
    return PreparedLesson.model_validate(prepared_payload())


def symbolic_route_and_steps():
    grounded = route()
    payload = grounded.to_prompt_payload()
    steps = [
        MathStep(
            purpose="整理参数关系",
            operation="simplify",
            state_before=["2n-2m+1=0"],
            state_after=["m-n=1/2"],
            reason="回到题目目标",
        )
    ]
    symbolic = _freeze_route(
        mode=TeachingRouteMode.SYMBOLIC_VERIFIED,
        consistency=TeachingRouteConsistency.CONSISTENT,
        method_name="代入法",
        final_conclusion=grounded.final_conclusion,
        assumptions=payload["assumptions"],
        steps=payload["steps"],
        check_evidence=payload["check_evidence"],
        symbolic_math_route_json="{}",
        symbolic_context={"math_steps": [item.model_dump() for item in steps]},
    )
    return symbolic, steps


def test_adapter_uses_script_clauses_as_the_only_explanatory_narration():
    prepared = approved_prepared()

    draft = prepared_lesson_to_draft(
        source_problem(), prepared, route(), verified_math_steps=None
    )

    authored = [clause.spoken_text for clause in prepared.teaching_script.clauses]
    adapted = [cue.spoken_text for cue in draft.opening_sync_cues]
    adapted.extend(cue.spoken_text for cue in draft.method_introduction_sync_cues)
    adapted.extend(
        cue.spoken_text for moment in draft.moments for cue in moment.sync_cues
    )
    adapted.extend(cue.spoken_text for cue in draft.summary_sync_cues)
    assert adapted == authored
    assert draft.opening == prepared.teaching_script.clauses[0].spoken_text
    assert draft.summary == prepared.teaching_script.clauses[-1].spoken_text


def test_adapter_unwraps_clause_actions_and_keeps_audio_empty():
    prepared = approved_prepared()
    draft = prepared_lesson_to_draft(source_problem(), prepared, route())
    lesson = LessonCompiler(lesson_id_factory=lambda: "prepared-lesson").compile(
        source_problem(), draft, {"review_status": "approved"}
    )

    expected_actions = {
        cue.cue_id: [
            bound.action.model_dump(exclude_none=True)
            for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions)
        ]
        for cue in prepared.performance_score.cues
    }
    runtime_cues = {
        cue.cue_id: cue
        for beat in lesson.beats
        for cue in beat.sync_cues
        if cue.cue_id in expected_actions
    }
    assert set(runtime_cues) == set(expected_actions)
    for cue_id, runtime_cue in runtime_cues.items():
        actual = [
            action.model_dump(exclude_none=True)
            for action in (
                *runtime_cue.lead_actions,
                *runtime_cue.start_actions,
                *runtime_cue.end_actions,
            )
        ]
        assert actual == expected_actions[cue_id]
        assert runtime_cue.audio_url is None
    assert all(beat.audio_url is None for beat in lesson.beats)


def test_adapter_binds_interaction_to_its_exact_authored_section_cue():
    prepared = approved_prepared()

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert all(moment.interaction is None for moment in draft.moments)
    assert len(draft.fixed_section_interactions_after_cue) == 1
    runtime = next(iter(draft.fixed_section_interactions_after_cue.values()))
    plan = prepared.interaction_plan.interactions[0]
    assert runtime.prompt == plan.prompt
    assert runtime.expected_answer == plan.correct_option_id
    assert runtime.hints == [plan.hint]
    assert runtime.explanation_after_correct == plan.correct_feedback
    assert [option.label for option in runtime.options] == [
        option.display_text for option in plan.options
    ]
    assert [option.feedback for option in runtime.options] == [
        plan.correct_feedback
        if option.option_id == plan.correct_option_id
        else plan.incorrect_feedback_by_option[option.option_id]
        for option in plan.options
    ]
    assert all(
        "canonical_answer" not in option.model_dump()
        and "correct" not in option.model_dump()
        for option in runtime.options
    )


def test_runtime_interaction_occurs_after_boundary_and_before_resume_clause():
    prepared = approved_prepared()
    lesson = LessonCompiler(lesson_id_factory=lambda: "boundary-order").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )

    events = []
    for beat in lesson.beats:
        events.extend(cue.cue_id for cue in beat.sync_cues)
        if beat.interaction is not None:
            events.append("interaction:%s" % beat.interaction.interaction_id)

    after = events.index("cue-clause-2")
    interaction = events.index("interaction:interaction-1")
    resume = events.index("cue-clause-2-resume")
    assert after < interaction < resume
    assert interaction == after + 1
    assert resume == interaction + 1


def test_adapter_allows_zero_interactions_and_merges_adjacent_free_episodes():
    payload = prepared_payload()
    payload["interaction_plan"]["interactions"] = []
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert all(moment.interaction is None for moment in draft.moments)
    assert len(draft.moments) < len(prepared.reasoning_trajectory.episodes)


def test_adapter_requires_symbolic_steps_only_for_symbolic_route():
    prepared = approved_prepared()
    symbolic, steps = symbolic_route_and_steps()

    with pytest.raises(ValueError, match="symbolic.*math steps"):
        prepared_lesson_to_draft(source_problem(), prepared, symbolic)
    with pytest.raises(ValueError, match="grounded.*math steps"):
        prepared_lesson_to_draft(
            source_problem(), prepared, route(), verified_math_steps=steps
        )

    draft = prepared_lesson_to_draft(
        source_problem(), prepared, symbolic, verified_math_steps=steps
    )
    assert draft.math_steps == steps
    assert draft.teaching_route["verification_mode"] == "symbolic_verified"
    assert draft.teaching_route["teaching_route_fingerprint"] == symbolic.fingerprint


def test_adapter_does_not_change_the_public_runtime_lesson_contract():
    before = set(RuntimeLesson.model_fields)
    lesson = LessonCompiler(lesson_id_factory=lambda: "contract-check").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), approved_prepared(), route()),
        {"review_status": "approved"},
    )

    assert set(lesson.model_dump(mode="json")) == before
    assert "episode_id" not in lesson.model_dump_json()
    assert "generation_record" not in lesson.model_dump_json()


def test_private_prepared_artifacts_reconstruct_episode_to_runtime_cues():
    prepared = approved_prepared()
    lesson = LessonCompiler(lesson_id_factory=lambda: "mapping-check").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )

    clause_episode = {
        clause.clause_id: clause.episode_id
        for clause in prepared.teaching_script.clauses
    }
    private_mapping = {
        episode.episode_id: [
            cue.cue_id
            for cue in prepared.performance_score.cues
            if any(
                clause_episode[clause_id] == episode.episode_id
                for clause_id in cue.clause_ids
            )
        ]
        for episode in prepared.reasoning_trajectory.episodes
    }
    runtime_cue_ids = {
        cue.cue_id for beat in lesson.beats for cue in beat.sync_cues
    }

    assert all(private_mapping.values())
    assert set().union(*map(set, private_mapping.values())).issubset(
        runtime_cue_ids
    )
    assert "episode_id" not in lesson.model_dump_json()


def test_adapter_rejects_a_prepared_lesson_that_is_not_approved():
    payload = prepared_payload()
    payload["review"]["status"] = "revision_required"
    payload["review"]["findings"][0]["severity"] = "material"
    prepared = PreparedLesson.model_validate(payload)

    with pytest.raises(ValueError, match="approved"):
        prepared_lesson_to_draft(source_problem(), prepared, route())


def test_overlay_enters_for_one_teaching_point_then_returns_to_base():
    payload = prepared_payload()
    payload["performance_score"] = overlay_score_payload()
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert [moment.layer for moment in draft.moments] == [
        "base",
        "comparison",
        "base",
    ]
    assert "".join(
        cue.spoken_text for moment in draft.moments for cue in moment.sync_cues
    ) == "".join(
        clause.spoken_text
        for clause in prepared.teaching_script.clauses
        if clause.clause_id
        not in {
            *prepared.teaching_script.opening_clause_ids,
            *prepared.teaching_script.method_introduction_clause_ids,
            *prepared.teaching_script.closing_summary_clause_ids,
        }
    )


def test_compiled_prepared_lesson_adds_only_fixed_runtime_navigation_speech():
    prepared = approved_prepared()
    lesson = LessonCompiler(lesson_id_factory=lambda: "speech-sources").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )

    explanatory = "".join(beat.narration for beat in lesson.beats[:-1])
    assert explanatory == "".join(
        clause.spoken_text for clause in prepared.teaching_script.clauses
    )
    assert lesson.beats[-1].narration == "现在换一道表面不同、结构相同的题。"
    runtime_interaction = next(
        beat.interaction
        for beat in lesson.beats
        if beat.interaction is not None
        and beat.interaction.interaction_id == "interaction-1"
    )
    planned = prepared.interaction_plan.interactions[0]
    assert runtime_interaction.prompt == planned.prompt
    assert (
        runtime_interaction.explanation_after_correct
        == planned.correct_feedback
    )
    assert [item.feedback for item in runtime_interaction.options] == [
        planned.correct_feedback,
        planned.incorrect_feedback_by_option["option-b"],
        planned.incorrect_feedback_by_option["option-c"],
    ]
    transfer = lesson.beats[-1].interaction
    transfer_plan = prepared.interaction_plan.transfer_item
    correct = next(
        option
        for option in transfer_plan.options
        if option.option_id == transfer_plan.correct_option_id
    )
    assert transfer.explanation_after_correct == ""
    assert transfer.correct_audio_url is None
    assert next(
        option.feedback
        for option in transfer.options
        if option.option_id == transfer.expected_answer
    ) == correct.feedback


def _merge_score_cues(payload, first_index, second_index):
    first = payload["performance_score"]["cues"][first_index]
    second = payload["performance_score"]["cues"].pop(second_index)
    first["clause_ids"].extend(second["clause_ids"])
    for phase in ("lead_actions", "start_actions", "end_actions"):
        first.setdefault(phase, []).extend(second.get(phase, []))


def test_adapter_splits_a_contiguous_cue_crossing_runtime_sections():
    payload = prepared_payload()
    _merge_score_cues(payload, 0, 1)
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())
    lesson = LessonCompiler(lesson_id_factory=lambda: "section-split").compile(
        source_problem(), draft, {"review_status": "approved"}
    )

    assert "".join(beat.narration for beat in lesson.beats[:-1]) == "".join(
        clause.spoken_text for clause in prepared.teaching_script.clauses
    )


def test_adapter_splits_a_contiguous_cue_crossing_adjacent_episodes():
    payload = prepared_payload()
    _merge_score_cues(payload, 2, 3)
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert "".join(
        cue.spoken_text for moment in draft.moments for cue in moment.sync_cues
    ) == "".join(
        clause.spoken_text
        for clause in prepared.teaching_script.clauses
        if clause.clause_id
        not in {
            *prepared.teaching_script.opening_clause_ids,
            *prepared.teaching_script.method_introduction_clause_ids,
            *prepared.teaching_script.closing_summary_clause_ids,
        }
    )


def test_adapter_splits_more_than_five_adjacent_free_cues_into_moments():
    payload = prepared_payload()
    payload["interaction_plan"]["interactions"] = []
    closing = payload["teaching_script"]["clauses"].pop()
    closing_cue = payload["performance_score"]["cues"].pop()
    for index in range(2):
        clause_id = "clause-6-extra-%d" % index
        payload["teaching_script"]["clauses"].append(
            {
                **payload["teaching_script"]["clauses"][-1],
                "clause_id": clause_id,
                "must_teach_refs": [],
                "spoken_text": "我们再检查一次当前关系。",
            }
        )
        payload["performance_score"]["cues"].append(
            {
                "cue_id": "cue-%s" % clause_id,
                "clause_ids": [clause_id],
                "start_actions": [
                    {
                        "clause_id": clause_id,
                        "action": {
                            "surface": "board",
                            "type": "focus",
                            "target": "board-6",
                        },
                    }
                ],
            }
        )
    payload["teaching_script"]["clauses"].append(closing)
    payload["performance_score"]["cues"].append(closing_cue)
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert len(draft.moments) >= 2
    assert all(len(moment.sync_cues) <= 5 for moment in draft.moments)
    assert "".join(
        cue.spoken_text for moment in draft.moments for cue in moment.sync_cues
    ) == "".join(
        clause.spoken_text
        for clause in prepared.teaching_script.clauses
        if clause.clause_id
        not in {
            *prepared.teaching_script.opening_clause_ids,
            *prepared.teaching_script.method_introduction_clause_ids,
            *prepared.teaching_script.closing_summary_clause_ids,
        }
    )
