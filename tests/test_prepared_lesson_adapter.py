import importlib.util
from dataclasses import FrozenInstanceError, replace

import pytest

import app.prepared_lesson_adapter as prepared_adapter
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
                        "target": "board-3",
                    },
                }
            ],
        },
    )
    interaction = payload["interaction_plan"]["interactions"][0]
    interaction.update(
        episode_id="episode-3",
        after_clause_id="clause-3",
        resume_clause_id="clause-3-resume",
    )
    if with_overlay:
        payload["performance_score"] = overlay_score_payload()
        resume_cue = cues[cue_index + 1]
        resume_cue["start_actions"] = []
        payload["performance_score"]["cues"].insert(
            cue_index + 1,
            resume_cue,
        )
    return payload


def later_clause_action_payload(phase):
    payload = prepared_payload()
    clauses = payload["teaching_script"]["clauses"]
    clause_index = next(
        index
        for index, clause in enumerate(clauses)
        if clause["clause_id"] == "clause-3"
    )
    clauses.insert(
        clause_index,
        {
            **clauses[clause_index],
            "clause_id": "clause-3-prelude",
            "spoken_text": "先停一下，观察我们已经得到的信息。",
            "must_teach_refs": [],
        },
    )
    cue = next(
        cue
        for cue in payload["performance_score"]["cues"]
        if cue["clause_ids"] == ["clause-3"]
    )
    cue["clause_ids"].insert(0, "clause-3-prelude")
    cue["lead_actions"] = []
    cue["start_actions"] = []
    cue["end_actions"] = []
    if phase == "start_actions":
        cue[phase] = [
            {
                "clause_id": "clause-3",
                "action": {
                    "surface": "board",
                    "type": "write",
                    "target": "board-3",
                    "content": payload["performance_score"][
                        "board_objects"
                    ][2]["content"],
                },
            }
        ]
    elif phase == "lead_actions":
        cue[phase] = [
            {
                "clause_id": "clause-3",
                "action": {
                    "surface": "board",
                    "type": "focus",
                    "target": "board-2",
                },
            }
        ]
        cue["end_actions"] = [
            {
                "clause_id": "clause-3",
                "action": {
                    "surface": "board",
                    "type": "clear_focus",
                    "target": "board-2",
                },
            }
        ]
    else:
        cue["lead_actions"] = [
            {
                "clause_id": "clause-3",
                "action": {
                    "surface": "board",
                    "type": "emphasize",
                    "target": "board-2",
                    "emphasis_style": "underline",
                },
            }
        ]
        cue[phase] = [
            {
                "clause_id": "clause-3",
                "action": {
                    "surface": "board",
                    "type": "fade",
                    "target": "board-2",
                },
            }
        ]
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
        if item.clause_id in {"clause-3-prelude", "clause-3"}
    ]
    prelude = next(
        item for item in records if item.clause_id == "clause-3-prelude"
    )
    action_clause = next(
        item for item in records if item.clause_id == "clause-3"
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
        "cue-clause-3"
    ]
    assert draft.moments[interaction_index - 1].sync_cues[-1].cue_id == (
        "cue-clause-2-resume"
    )
    assert draft.moments[interaction_index + 1].sync_cues[0].cue_id == (
        "cue-clause-3-resume"
    )


def test_overlay_interaction_preserves_authored_comparison_layer():
    prepared = PreparedLesson.model_validate(
        body_interaction_payload(with_overlay=True)
    )

    draft = prepared_lesson_to_draft(source_problem(), prepared, route())

    interaction_moment = next(
        moment for moment in draft.moments if moment.interaction is not None
    )
    assert interaction_moment.layer == "comparison"
    assert [cue.cue_id for cue in interaction_moment.sync_cues] == [
        "cue-clause-3"
    ]


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


def test_adapter_carries_authored_base_layer_for_fixed_method_clause():
    draft = prepared_lesson_to_draft(
        source_problem(), approved_prepared(), route()
    )

    method_cue_id = draft.method_introduction_sync_cues[0].cue_id
    assert draft.fixed_section_layers_by_cue[method_cue_id] == "base"


def test_adapter_carries_comparison_layer_for_fixed_summary_clause():
    payload = prepared_payload()
    payload["performance_score"]["board_objects"][6]["layer"] = (
        "comparison"
    )
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
    assert {item.runtime_cue_id for item in run.cue_provenance} == set(
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
    assert [item.clause_id for item in rebuilt.cue_provenance] == [
        clause.clause_id for clause in authoritative_clauses
    ]
    assert [item.episode_id for item in rebuilt.cue_provenance] == [
        clause.episode_id for clause in authoritative_clauses
    ]
    assert [item.spoken_text for item in rebuilt.cue_provenance] == [
        clause.spoken_text for clause in authoritative_clauses
    ]
    assert [
        item.original_performance_cue_id
        for item in rebuilt.cue_provenance
    ] == [
        authoritative_original[clause.clause_id]
        for clause in authoritative_clauses
    ]
    assert rebuilt.draft.title != changed.title


def grouped_provenance_run():
    payload = later_clause_action_payload("start_actions")
    cue = next(
        cue
        for cue in payload["performance_score"]["cues"]
        if cue["clause_ids"] == ["clause-3-prelude", "clause-3"]
    )
    cue["lead_actions"] = []
    cue["start_actions"] = []
    cue["end_actions"] = []
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
            if item.clause_id == "clause-3-prelude"
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
        item.clause_id: item.runtime_cue_id for item in run.cue_provenance
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
        item.clause_id: item.runtime_cue_id for item in run.cue_provenance
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
    payload["teaching_script"]["clauses"][1]["spoken_text"] = (
        "再说明为什么可以代入。"
    )
    prepared = PreparedLesson.model_validate(payload)
    run = prepared_adapter.prepared_lesson_to_draft_with_provenance(
        source_problem(), prepared, route()
    )
    assignment = {
        item.clause_id: item.runtime_cue_id for item in run.cue_provenance
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
    assert {item.runtime_cue_id for item in run.cue_provenance} == set(
        runtime_cues
    )
    assert all(
        runtime_cues[item.runtime_cue_id].spoken_text == item.spoken_text
        for item in split
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
