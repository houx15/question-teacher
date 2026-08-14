import importlib.util
import json
from dataclasses import FrozenInstanceError, replace

import pytest

import app.prepared_lesson_adapter as prepared_adapter
from app.api import public_lesson_payload
from app.compiler import LessonCompiler
from app.preparation_models import PreparedLesson
from app.prepared_lesson_adapter import (
    PreparedLessonAdaptationError,
    prepared_lesson_to_draft,
)
from app.schemas import MathStep, ProblemInput, RuntimeLesson
from app.teaching_route import (
    TeachingRouteConsistency,
    TeachingRouteMode,
    _freeze_route,
)
from tests.test_preparation_validation import (
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


def body_interaction_payload(with_overlay=False):
    payload = prepared_payload()
    clauses = payload["teaching_script"]["clauses"]
    clause_index = next(
        index
        for index, clause in enumerate(clauses)
        if clause["clause_id"] == "clause-3"
    )
    clauses.insert(
        clause_index + 1,
        {
            **clauses[clause_index],
            "clause_id": "clause-3-resume",
            "spoken_text": "用刚才的判断继续处理这一步。",
            "pedagogical_function": "transition",
            "must_teach_refs": [],
        },
    )
    cues = payload["performance_score"]["cues"]
    cue_index = next(
        index
        for index, cue in enumerate(cues)
        if cue["clause_ids"] == ["clause-3"]
    )
    cues.insert(
        cue_index + 1,
        {
            "cue_id": "cue-clause-3-resume",
            "clause_ids": ["clause-3-resume"],
            "start_actions": [
                {
                    "clause_id": "clause-3-resume",
                    "action": {
                        "surface": "board",
                        "type": "focus",
                        "target": "board-clause-3",
                        "teaching_step_id": "teaching-step-3",
                    },
                }
            ],
            "end_actions": [
                {
                    "clause_id": "clause-3-resume",
                    "action": {
                        "surface": "board",
                        "type": "complete_step",
                        "target": "teaching-step-3",
                        "teaching_step_id": "teaching-step-3",
                    },
                }
            ],
        },
    )
    original_step_cue = cues[cue_index]
    original_step_cue["end_actions"] = [
        item
        for item in original_step_cue["end_actions"]
        if item["action"]["type"] != "complete_step"
    ]
    interaction = payload["interaction_plan"]["interactions"][0]
    interaction.update(
        episode_id="episode-3",
        teaching_step_id="teaching-step-3",
        after_clause_id="clause-3",
        why_pause="停下检查学生是否知道如何处理当前学习检查点。",
        resume_clause_id="clause-3-resume",
        resume_step_id="teaching-step-3",
    )
    payload["teaching_progression"]["steps"][2]["checkpoint"] = {
        "diagnostic_goal": "检查学生是否知道如何处理当前学习检查点",
        "misconception_ids": [],
    }
    for response in payload["teaching_script"]["response_scripts"]:
        if response["interaction_id"] == interaction["interaction_id"]:
            for clause in response["clauses"]:
                clause["episode_id"] = "episode-3"
                clause["lesson_step_id"] = "teaching-step-3"
    response_clause_ids = {
        clause["clause_id"]
        for response in payload["teaching_script"]["response_scripts"]
        for clause in response["clauses"]
    }
    for cue in payload["performance_score"]["cues"]:
        if not response_clause_ids.intersection(cue["clause_ids"]):
            continue
        for phase in ("lead_actions", "start_actions", "end_actions"):
            for bound in cue.get(phase, []):
                action = bound["action"]
                action["teaching_step_id"] = "teaching-step-3"
                if action["type"] == "scroll_to_step":
                    action["target"] = "teaching-step-3"
    for board_object in payload["performance_score"]["board_objects"]:
        if board_object.get("line_role") == "support":
            board_object["teaching_step_id"] = "teaching-step-3"
    if with_overlay:
        score = payload["performance_score"]
        next(
            item
            for item in score["board_objects"]
            if item["board_object_id"] == "board-clause-3"
        )["layer"] = "comparison"
        score["overlay_transitions"] = [
            {
                "transition_id": "enter-comparison",
                "after_clause_id": "clause-2-resume",
                "action": "enter",
                "layer": "comparison",
            },
            {
                "transition_id": "return-comparison",
                "after_clause_id": "clause-3",
                "action": "return",
                "layer": "comparison",
            },
        ]
        next(
            cue
            for cue in score["cues"]
            if cue["clause_ids"] == ["clause-3-resume"]
        )["start_actions"] = []
    return payload


def later_clause_action_payload(phase):
    payload = prepared_payload()
    clauses = payload["teaching_script"]["clauses"]
    clause_index = next(
        index
        for index, clause in enumerate(clauses)
        if clause["clause_id"] == "clause-2-resume"
    )
    clauses.insert(
        clause_index,
        {
            **clauses[clause_index],
            "clause_id": "clause-2-prelude",
            "spoken_text": "先停一下，观察我们已经得到的信息。",
            "must_teach_refs": [],
        },
    )
    cue = next(
        cue
        for cue in payload["performance_score"]["cues"]
        if cue["clause_ids"] == ["clause-2-resume"]
    )
    cue["clause_ids"].insert(0, "clause-2-prelude")
    step_id = "teaching-step-2"
    target = "board-clause-2"
    if phase == "start_actions":
        cue.setdefault(phase, []).append(
            {
                "clause_id": "clause-2-resume",
                "action": {
                    "surface": "board",
                    "type": "focus",
                    "target": target,
                    "teaching_step_id": step_id,
                },
            }
        )
    elif phase == "lead_actions":
        cue.setdefault(phase, []).append(
            {
                "clause_id": "clause-2-resume",
                "action": {
                    "surface": "board",
                    "type": "focus",
                    "target": target,
                    "teaching_step_id": step_id,
                },
            }
        )
        cue.setdefault("end_actions", []).append(
            {
                "clause_id": "clause-2-resume",
                "action": {
                    "surface": "board",
                    "type": "clear_focus",
                    "target": target,
                    "teaching_step_id": step_id,
                },
            }
        )
    else:
        cue.setdefault("lead_actions", []).append(
            {
                "clause_id": "clause-2-resume",
                "action": {
                    "surface": "board",
                    "type": "emphasize",
                    "target": target,
                    "emphasis_style": "underline",
                    "teaching_step_id": step_id,
                },
            }
        )
        cue.setdefault(phase, []).append(
            {
                "clause_id": "clause-2-resume",
                "action": {
                    "surface": "board",
                    "type": "fade",
                    "target": target,
                    "teaching_step_id": step_id,
                },
            }
        )
    return payload


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
        target_expression=grounded.target_expression,
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


def test_body_moment_purposes_use_verified_math_operations_not_designer_prose():
    payload = prepared_payload()
    for episode in payload["reasoning_trajectory"]["episodes"]:
        episode["decision"] = "Substitute private designer prose"
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(
        source_problem(), prepared, route(), verified_math_steps=None
    )

    purposes = [moment.purpose for moment in draft.moments]
    assert purposes
    assert all(
        "Substitute private designer prose" not in item
        for item in purposes
    )
    assert purposes[0] == "代入已知数学量"
    assert all(
        any("\u4e00" <= char <= "\u9fff" for char in item)
        for item in purposes
    )


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
        if all(
            clause_id
            in {
                clause.clause_id
                for clause in prepared.teaching_script.clauses
            }
            for clause_id in cue.clause_ids
        )
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


@pytest.mark.parametrize(
    "phase",
    ("lead_actions", "start_actions", "end_actions"),
)
def test_later_clause_actions_keep_their_own_runtime_timing_boundary(phase):
    prepared = PreparedLesson.model_validate(
        later_clause_action_payload(phase)
    )

    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    lesson = LessonCompiler(lesson_id_factory=lambda: "action-timing").compile(
        source_problem(), run.draft, {"review_status": "approved"}
    )
    runtime_cues = {
        cue.cue_id: cue
        for beat in lesson.beats
        for cue in beat.sync_cues
    }
    records = [
        item
        for item in run.cue_provenance
        if item.clause_id in {"clause-2-prelude", "clause-2-resume"}
    ]
    prelude = next(
        item for item in records if item.clause_id == "clause-2-prelude"
    )
    action_clause = next(
        item for item in records if item.clause_id == "clause-2-resume"
    )

    assert prelude.runtime_cue_id != action_clause.runtime_cue_id
    assert runtime_cues[prelude.runtime_cue_id].spoken_text == prelude.spoken_text
    assert runtime_cues[action_clause.runtime_cue_id].spoken_text == (
        action_clause.spoken_text
    )
    assert runtime_cues[prelude.runtime_cue_id].lead_actions == []
    assert runtime_cues[prelude.runtime_cue_id].start_actions == []
    assert runtime_cues[prelude.runtime_cue_id].end_actions == []
    assert getattr(runtime_cues[action_clause.runtime_cue_id], phase)
    assert "".join(
        cue.spoken_text
        for beat in lesson.beats[:-1]
        for cue in beat.sync_cues
    ) == "".join(
        clause.spoken_text for clause in prepared.teaching_script.clauses
    )


def test_adapter_binds_interaction_to_its_exact_authored_section_cue():
    prepared = approved_prepared()

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    runtimes = [
        *draft.fixed_section_interactions_after_cue.values(),
        *(
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ),
    ]
    assert len(runtimes) == 1
    runtime = runtimes[0]
    plan = prepared.interaction_plan.interactions[0]
    assert runtime.prompt == plan.prompt
    assert runtime.expected_answer == plan.correct_option_id
    assert runtime.hints == [plan.hint]
    assert runtime.explanation_after_correct == ""
    assert runtime.advance_after_response is True
    assert [option.label for option in runtime.options] == [
        option.display_text for option in plan.options
    ]
    assert all(option.feedback is None for option in runtime.options)
    response_by_option = {
        response.option_id: response
        for response in prepared.teaching_script.response_scripts
    }
    performance_by_clause = {
        clause_id: cue
        for cue in prepared.performance_score.cues
        for clause_id in cue.clause_ids
    }
    for option in runtime.options:
        response = response_by_option[option.option_id]
        assert [cue.cue_id for cue in option.support_cues] == [
            clause.clause_id for clause in response.clauses
        ]
        assert [cue.display_text for cue in option.support_cues] == [
            clause.display_text for clause in response.clauses
        ]
        assert [cue.spoken_text for cue in option.support_cues] == [
            clause.spoken_text for clause in response.clauses
        ]
        assert all(cue.audio_url is None for cue in option.support_cues)
        for cue, clause in zip(option.support_cues, response.clauses):
            performance_cue = performance_by_clause[clause.clause_id]
            assert cue.lead_actions == [
                item.action
                for item in performance_cue.lead_actions
                if item.clause_id == clause.clause_id
            ]
            assert cue.start_actions == [
                item.action
                for item in performance_cue.start_actions
                if item.clause_id == clause.clause_id
            ]
            assert cue.end_actions == [
                item.action
                for item in performance_cue.end_actions
                if item.clause_id == clause.clause_id
            ]
    assert all(
        "canonical_answer" not in option.model_dump()
        and "correct" not in option.model_dump()
        for option in runtime.options
    )


def test_authoritative_adapter_factory_provenance_covers_main_and_response_clauses():
    prepared = approved_prepared()
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )

    main = prepared.teaching_script.clauses
    responses = [
        (response, clause)
        for response in prepared.teaching_script.response_scripts
        for clause in response.clauses
    ]
    assert [item.clause_id for item in run.cue_provenance] == [
        *[clause.clause_id for clause in main],
        *[clause.clause_id for _response, clause in responses],
    ]
    for item, clause in zip(run.cue_provenance[: len(main)], main):
        assert item.lesson_step_id == clause.lesson_step_id
        assert item.display_text == clause.display_text
        assert item.response_id is None
    for item, (response, clause) in zip(
        run.cue_provenance[len(main) :], responses
    ):
        assert item.lesson_step_id == clause.lesson_step_id
        assert item.display_text == clause.display_text
        assert item.response_id == response.response_id
        assert item.runtime_cue_id == clause.clause_id


def test_authoritative_adapter_factory_rejects_swapped_support_binding():
    prepared = approved_prepared()
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    draft = run.draft
    interaction = [
        *draft.fixed_section_interactions_after_cue.values(),
        *(
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ),
    ][0]
    first, second = interaction.options[:2]
    first.support_cues, second.support_cues = (
        second.support_cues,
        first.support_cues,
    )
    main_assignment = {
        item.clause_id: item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    }

    with pytest.raises(ValueError, match="response binding"):
        prepared_adapter.PreparedDraftRun.from_prepared_lesson(
            draft,
            prepared,
            main_assignment,
        )


def test_runtime_interaction_occurs_after_current_teaching_step():
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
    assert after < resume < interaction
    assert resume == interaction - 1


def test_structured_interaction_derives_boundary_without_legacy_clause_ids():
    payload = prepared_payload()
    interaction = payload["interaction_plan"]["interactions"][0]
    interaction.pop("after_clause_id")
    interaction.pop("resume_clause_id")
    prepared = PreparedLesson.model_validate(payload)

    lesson = LessonCompiler(lesson_id_factory=lambda: "structured-boundary").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )

    runtime = next(
        beat.interaction
        for beat in lesson.beats
        if beat.interaction is not None
        and beat.interaction.interaction_id == "interaction-1"
    )
    assert runtime.advance_after_response is True
    assert all(option.support_cues for option in runtime.options)


def test_structured_interaction_ignores_stale_legacy_clause_boundary():
    payload = prepared_payload()
    payload["interaction_plan"]["interactions"][0][
        "after_clause_id"
    ] = "clause-1"
    prepared = PreparedLesson.model_validate(payload)

    lesson = LessonCompiler(lesson_id_factory=lambda: "structured-step").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )
    events = []
    for beat in lesson.beats:
        events.extend(cue.cue_id for cue in beat.sync_cues)
        if beat.interaction is not None:
            events.append("interaction:%s" % beat.interaction.interaction_id)

    assert events.index("cue-clause-2-resume") < events.index(
        "interaction:interaction-1"
    )


def test_structured_runtime_public_payload_contains_support_without_private_binding():
    prepared = approved_prepared()
    lesson = LessonCompiler(lesson_id_factory=lambda: "public-support").compile(
        source_problem(),
        prepared_lesson_to_draft(source_problem(), prepared, route()),
        {"review_status": "approved"},
    )

    payload = public_lesson_payload(lesson)
    serialized = json.dumps(payload, ensure_ascii=False)
    interaction = next(
        beat["interaction"]
        for beat in payload["beats"]
        if (beat.get("interaction") or {}).get("interaction_id")
        == "interaction-1"
    )

    assert interaction["advance_after_response"] is True
    assert all("support_cues" not in option for option in interaction["options"])
    for private_field in (
        "expected_answer",
        "canonical_answer",
        "correct_option_id",
        "error_code",
        "remediation_depth",
        "misconception",
    ):
        assert private_field not in serialized


def test_body_interaction_episode_is_not_merged_with_preceding_free_episode():
    prepared = PreparedLesson.model_validate(body_interaction_payload())

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    interaction_index = next(
        index
        for index, moment in enumerate(draft.moments)
        if moment.interaction is not None
    )
    interaction_moment = draft.moments[interaction_index]
    assert [cue.cue_id for cue in interaction_moment.sync_cues] == [
        "cue-clause-3",
        "cue-clause-3-resume",
    ]
    assert draft.moments[interaction_index - 1].sync_cues[-1].cue_id == (
        "cue-clause-2-resume"
    )
    assert draft.moments[interaction_index + 1].sync_cues[0].cue_id == (
        "cue-clause-4"
    )


def test_step_end_interaction_uses_the_authored_boundary_layer():
    prepared = PreparedLesson.model_validate(
        body_interaction_payload(with_overlay=True)
    )

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    interaction_moment = next(
        moment for moment in draft.moments if moment.interaction is not None
    )
    assert interaction_moment.layer == "base"
    assert [cue.cue_id for cue in interaction_moment.sync_cues] == [
        "cue-clause-3-resume",
    ]


def test_adapter_allows_zero_interactions_and_merges_adjacent_free_episodes():
    payload = prepared_payload()
    payload["interaction_plan"]["interactions"] = []
    payload["teaching_script"]["response_scripts"] = []
    payload["teaching_script"]["interaction_scripts"] = []
    payload["performance_score"]["cues"] = [
        cue
        for cue in payload["performance_score"]["cues"]
        if not cue["clause_ids"][0].startswith("response-clause-")
    ]
    payload["performance_score"]["board_objects"] = [
        item
        for item in payload["performance_score"]["board_objects"]
        if item.get("line_role") != "support"
    ]
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    assert all(moment.interaction is None for moment in draft.moments)
    assert len(draft.moments) < len(prepared.reasoning_trajectory.episodes)


def test_legacy_public_marker_in_private_plan_never_reaches_current_runtime():
    payload = prepared_payload()
    marker = "PRIVATE-PLAN-PUBLIC-DRAFT-MARKER"
    interaction = payload["interaction_plan"]["interactions"][0]
    interaction["prompt"] = "Correct answer: option-a %s" % marker
    interaction["hint"] = "正确答案是代入已知根 %s" % marker
    interaction["options"][1]["display_text"] = (
        "内部草稿提及正确答案：代入已知根 %s" % marker
    )
    payload["interaction_plan"]["transfer_item"]["problem_text"] = marker

    draft = prepared_lesson_to_draft(
        source_problem(), PreparedLesson.model_validate(payload), route()
    )
    lesson = LessonCompiler(lesson_id_factory=lambda: "private-intent").compile(
        source_problem(),
        draft,
        {"review_status": "approved"},
    )
    public_payload = public_lesson_payload(lesson)

    assert marker not in draft.model_dump_json()
    assert marker not in json.dumps(public_payload, ensure_ascii=False)


def test_adapter_requires_symbolic_steps_only_for_symbolic_route():
    prepared = approved_prepared()
    symbolic, steps = symbolic_route_and_steps()

    with pytest.raises(PreparedLessonAdaptationError):
        prepared_lesson_to_draft(source_problem(), prepared, symbolic)
    with pytest.raises(PreparedLessonAdaptationError):
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
            if all(clause_id in clause_episode for clause_id in cue.clause_ids)
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

    with pytest.raises(PreparedLessonAdaptationError):
        prepared_lesson_to_draft(source_problem(), prepared, route())


def test_overlay_enters_for_one_teaching_point_then_returns_to_base():
    payload = prepared_payload()
    score = payload["performance_score"]
    next(
        item
        for item in score["board_objects"]
        if item["board_object_id"] == "board-clause-3"
    )["layer"] = "comparison"
    score["overlay_transitions"] = [
        {
            "transition_id": "enter-comparison",
            "after_clause_id": "clause-2-resume",
            "action": "enter",
            "layer": "comparison",
        },
        {
            "transition_id": "return-comparison",
            "after_clause_id": "clause-3",
            "action": "return",
            "layer": "comparison",
        },
    ]
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


def test_adapter_carries_authored_base_layer_for_fixed_method_clause():
    draft = prepared_lesson_to_draft(
        source_problem(), approved_prepared(), route()
    )

    method_cue_id = draft.method_introduction_sync_cues[0].cue_id
    assert draft.fixed_section_layers_by_cue[method_cue_id] == "base"


def test_adapter_carries_comparison_layer_for_fixed_summary_clause():
    payload = prepared_payload()
    next(
        item
        for item in payload["performance_score"]["board_objects"]
        if item["board_object_id"] == "board-clause-7"
    )["layer"] = "comparison"
    payload["performance_score"]["overlay_transitions"] = [
        {
            "transition_id": "enter-summary-comparison",
            "after_clause_id": "clause-6",
            "action": "enter",
            "layer": "comparison",
        },
        {
            "transition_id": "return-summary-comparison",
            "after_clause_id": "clause-7",
            "action": "return",
            "layer": "comparison",
        },
    ]
    prepared = PreparedLesson.model_validate(payload)

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())
    lesson = LessonCompiler(lesson_id_factory=lambda: "summary-layer").compile(
        source_problem(), draft, {"review_status": "approved"}
    )

    summary = next(beat for beat in lesson.beats if beat.purpose == "压缩方法")
    assert summary.layer == "comparison"


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
    assert runtime_interaction.explanation_after_correct == ""
    assert runtime_interaction.advance_after_response is True
    assert all(
        item.feedback is None and item.support_cues
        for item in runtime_interaction.options
    )
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


def test_cross_section_split_provenance_maps_every_runtime_cue_exactly():
    payload = prepared_payload()
    original_cue_id = payload["performance_score"]["cues"][0]["cue_id"]
    _merge_score_cues(payload, 0, 1)
    prepared = PreparedLesson.model_validate(payload)

    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    lesson = LessonCompiler(lesson_id_factory=lambda: "section-provenance").compile(
        source_problem(), run.draft, {"review_status": "approved"}
    )

    runtime_cues = {
        cue.cue_id: cue
        for beat in lesson.beats
        for cue in beat.sync_cues
        if cue.cue_id != "runtime-transfer-intro-cue"
    }
    assert {
        item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    } == set(
        runtime_cues
    )
    split = [
        item
        for item in run.cue_provenance
        if item.original_performance_cue_id == original_cue_id
    ]
    assert [item.clause_id for item in split] == ["clause-1", "clause-2"]
    assert split[0].runtime_cue_id == original_cue_id
    assert split[1].runtime_cue_id.startswith("prepared-cue-")
    assert all(
        runtime_cues[item.runtime_cue_id].spoken_text == item.spoken_text
        for item in split
    )


def test_prepared_draft_run_is_defensive_and_provenance_is_immutable():
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), approved_prepared(), route()
    )

    changed = run.draft
    changed.title = "被调用方修改"

    assert run.draft.title != changed.title
    assert isinstance(run.cue_provenance, tuple)
    with pytest.raises(FrozenInstanceError):
        run.cue_provenance[0].runtime_cue_id = "changed"


def test_prepared_draft_factory_derives_records_without_mutable_alias():
    prepared = approved_prepared()
    original = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    runtime_cue_by_clause = {
        item.clause_id: item.runtime_cue_id
        for item in original.cue_provenance
        if item.response_id is None
    }

    rebuilt = prepared_adapter.PreparedDraftRun.from_prepared_lesson(
        original.draft,
        prepared,
        runtime_cue_by_clause,
    )
    runtime_cue_by_clause.clear()
    changed = rebuilt.draft
    changed.title = "外部修改"

    authoritative_clauses = prepared.teaching_script.clauses
    authoritative_original = {
        clause_id: cue.cue_id
        for cue in prepared.performance_score.cues
        for clause_id in cue.clause_ids
    }
    rebuilt_main = [
        item for item in rebuilt.cue_provenance if item.response_id is None
    ]
    assert [item.clause_id for item in rebuilt_main] == [
        clause.clause_id for clause in authoritative_clauses
    ]
    assert [item.episode_id for item in rebuilt_main] == [
        clause.episode_id for clause in authoritative_clauses
    ]
    assert [item.spoken_text for item in rebuilt_main] == [
        clause.spoken_text for clause in authoritative_clauses
    ]
    assert [
        item.original_performance_cue_id
        for item in rebuilt_main
    ] == [
        authoritative_original[clause.clause_id]
        for clause in authoritative_clauses
    ]
    assert rebuilt.draft.title != changed.title


def grouped_provenance_run():
    payload = prepared_payload()
    clauses = payload["teaching_script"]["clauses"]
    resume_index = next(
        index
        for index, clause in enumerate(clauses)
        if clause["clause_id"] == "clause-2-resume"
    )
    grouped_clauses = [
        {
            **clauses[resume_index],
            "clause_id": "clause-2-grouped-%d" % index,
            "spoken_text": "我们补充观察当前条件。",
            "must_teach_refs": [],
        }
        for index in range(2)
    ]
    clauses[resume_index:resume_index] = grouped_clauses
    cues = payload["performance_score"]["cues"]
    resume_cue_index = next(
        index
        for index, cue in enumerate(cues)
        if cue["clause_ids"] == ["clause-2-resume"]
    )
    cues.insert(
        resume_cue_index,
        {
            "cue_id": "cue-clause-2-grouped",
            "clause_ids": [
                clause["clause_id"] for clause in grouped_clauses
            ],
        },
    )
    prepared = PreparedLesson.model_validate(payload)
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    return prepared, run


def test_prepared_draft_run_zero_argument_constructor_names_factory():
    with pytest.raises(TypeError, match="^use from_prepared_lesson$"):
        prepared_adapter.PreparedDraftRun()


@pytest.mark.parametrize(
    "attack",
    ("unchanged", "episode_id", "original_cue_id", "text_repartition"),
)
def test_prepared_draft_run_direct_constructor_cannot_inject_provenance(
    attack,
):
    prepared = approved_prepared()
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    records = list(run.cue_provenance)
    if attack == "episode_id":
        records[0] = replace(
            records[0],
            episode_id=records[-1].episode_id,
        )
    elif attack == "original_cue_id":
        records[0] = replace(
            records[0],
            original_performance_cue_id=(
                records[-1].original_performance_cue_id
            ),
        )
    elif attack == "text_repartition":
        prepared, run = grouped_provenance_run()
        records = list(run.cue_provenance)
        first_index = next(
            index
            for index, item in enumerate(records)
            if item.clause_id == "clause-2-grouped-0"
        )
        second_index = first_index + 1
        first = records[first_index]
        second = records[second_index]
        records[first_index] = replace(
            first,
            spoken_text=first.spoken_text + second.spoken_text[0],
        )
        records[second_index] = replace(
            second,
            spoken_text=second.spoken_text[1:],
        )
    expected = tuple(
        clause.clause_id for clause in prepared.teaching_script.clauses
    )

    with pytest.raises(TypeError, match="^use from_prepared_lesson$"):
        prepared_adapter.PreparedDraftRun(run.draft, records, expected)


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "runtime_id"),
)
def test_prepared_draft_factory_rejects_invalid_runtime_assignment(mutation):
    prepared = approved_prepared()
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    assignment = {
        item.clause_id: item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    }
    if mutation == "missing":
        assignment.pop(next(iter(assignment)))
        message = "complete clause mapping"
    elif mutation == "extra":
        assignment["forged-clause"] = next(iter(assignment.values()))
        message = "complete clause mapping"
    else:
        assignment[next(iter(assignment))] = "forged-runtime-cue"
        message = "runtime cue ids"

    with pytest.raises(ValueError, match=message):
        prepared_adapter.PreparedDraftRun.from_prepared_lesson(
            run.draft,
            prepared,
            assignment,
        )


@pytest.mark.parametrize("membership", ("missing", "duplicate"))
def test_prepared_draft_factory_rejects_invalid_performance_membership(
    membership,
):
    prepared = approved_prepared()
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    payload = prepared.model_dump(mode="python")
    if membership == "missing":
        payload["performance_score"]["cues"].pop(0)
    else:
        duplicate_clause_id = payload["performance_score"]["cues"][0][
            "clause_ids"
        ][0]
        payload["performance_score"]["cues"][1]["clause_ids"].append(
            duplicate_clause_id
        )
    invalid = PreparedLesson.model_validate(payload)
    assignment = {
        item.clause_id: item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    }

    with pytest.raises(ValueError, match="performance cue membership"):
        prepared_adapter.PreparedDraftRun.from_prepared_lesson(
            run.draft,
            invalid,
            assignment,
        )


def test_prepared_draft_factory_rejects_grouped_runtime_text_mismatch():
    payload = prepared_payload()
    payload["teaching_script"]["clauses"][0]["spoken_text"] = (
        "先观察题目中的目标。"
    )
    payload["reasoning_trajectory"]["episodes"][0]["must_teach"][0][
        "student_spoken_evidence"
    ] = "先观察题目中的目标。"
    payload["teaching_script"]["clauses"][1]["spoken_text"] = (
        "再说明为什么可以代入。"
    )
    payload["reasoning_trajectory"]["episodes"][1]["must_teach"][0][
        "student_spoken_evidence"
    ] = "再说明为什么可以代入。"
    prepared = PreparedLesson.model_validate(payload)
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    assignment = {
        item.clause_id: item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    }
    first_id = prepared.teaching_script.clauses[0].clause_id
    second_id = prepared.teaching_script.clauses[1].clause_id
    assignment[first_id], assignment[second_id] = (
        assignment[second_id],
        assignment[first_id],
    )

    with pytest.raises(ValueError, match="grouped provenance text"):
        prepared_adapter.PreparedDraftRun.from_prepared_lesson(
            run.draft,
            prepared,
            assignment,
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


def test_cross_episode_split_provenance_keeps_original_and_generated_ids():
    payload = prepared_payload()
    original_cue_id = payload["performance_score"]["cues"][2]["cue_id"]
    _merge_score_cues(payload, 2, 3)
    prepared = PreparedLesson.model_validate(payload)

    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    lesson = LessonCompiler(lesson_id_factory=lambda: "episode-provenance").compile(
        source_problem(), run.draft, {"review_status": "approved"}
    )
    runtime_cues = {
        cue.cue_id: cue
        for beat in lesson.beats
        for cue in beat.sync_cues
        if cue.cue_id != "runtime-transfer-intro-cue"
    }
    split = [
        item
        for item in run.cue_provenance
        if item.original_performance_cue_id == original_cue_id
    ]

    assert [item.episode_id for item in split] == ["episode-2", "episode-3"]
    assert split[0].runtime_cue_id == original_cue_id
    assert split[1].runtime_cue_id.startswith("prepared-cue-")
    assert {
        item.runtime_cue_id
        for item in run.cue_provenance
        if item.response_id is None
    } == set(
        runtime_cues
    )
    assert all(
        runtime_cues[item.runtime_cue_id].spoken_text == item.spoken_text
        for item in split
    )


def test_adapter_splits_more_than_five_adjacent_free_cues_into_moments():
    payload = prepared_payload()
    payload["interaction_plan"]["interactions"] = []
    payload["teaching_script"]["response_scripts"] = []
    payload["teaching_script"]["interaction_scripts"] = []
    payload["performance_score"]["cues"] = [
        cue
        for cue in payload["performance_score"]["cues"]
        if not cue["clause_ids"][0].startswith("response-clause-")
    ]
    payload["performance_score"]["board_objects"] = [
        item
        for item in payload["performance_score"]["board_objects"]
        if item.get("line_role") != "support"
    ]
    closing = payload["teaching_script"]["clauses"].pop()
    closing_cue_index = next(
        index
        for index, cue in enumerate(payload["performance_score"]["cues"])
        if cue["clause_ids"] == [closing["clause_id"]]
    )
    closing_cue = payload["performance_score"]["cues"].pop(
        closing_cue_index
    )
    clause_six_cue = next(
        cue
        for cue in payload["performance_score"]["cues"]
        if cue["clause_ids"] == ["clause-6"]
    )
    clause_six_cue["end_actions"] = [
        item
        for item in clause_six_cue["end_actions"]
        if item["action"]["type"] != "complete_step"
    ]
    for index in range(2):
        clause_id = "clause-6-extra-%d" % index
        payload["teaching_script"]["clauses"].append(
            {
                **payload["teaching_script"]["clauses"][-1],
                "clause_id": clause_id,
                "must_teach_refs": [],
                "spoken_text": "我们再检查一次当前关系。",
                "pedagogical_function": "review",
            }
        )
        extra_cue = {
            "cue_id": "cue-%s" % clause_id,
            "clause_ids": [clause_id],
        }
        if index == 1:
            extra_cue["end_actions"] = [
                {
                    "clause_id": clause_id,
                    "action": {
                        "surface": "board",
                        "type": "complete_step",
                        "target": "teaching-step-6",
                        "teaching_step_id": "teaching-step-6",
                    },
                }
            ]
        payload["performance_score"]["cues"].append(extra_cue)
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
