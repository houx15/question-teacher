import asyncio
import copy
import json

import pytest

from app.generation import (
    LessonGenerationService,
    LessonQualityError,
    _VerifiedMathRoute,
)
from app.llm_client import ModelResponseError
from app.math_engine import MathEngine
from app.prompts import (
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REVIEWER_SYSTEM,
    REVISION_SYSTEM,
    director_prompt,
    materials_prompt,
)
from app.schemas import MaterialsDraft, MathRouteDraft, NarrativeDraft
from tests.generation_fakes import FakeClient
from tests.test_generation import (
    approved_review,
    approved_audit,
    problem,
    revision_review,
    valid_draft,
)


def narrative_payload():
    payload = copy.deepcopy(valid_draft())
    payload.pop("transfer_item")
    payload.pop("math_steps")
    for index, moment in enumerate(payload["moments"]):
        moment["moment_id"] = f"moment-{index}"
        if moment.get("interaction") is not None:
            moment["interaction_intent"] = (
                "诊断学生能否同时检查乘积和一次项系数。"
            )
        moment.pop("interaction", None)
        if moment.get("layer") == "interaction":
            moment["layer"] = "base"
    return payload


def materials_payload():
    draft = valid_draft()
    transfer_item = copy.deepcopy(draft["transfer_item"])
    for option in transfer_item["options"]:
        option.pop("label", None)
    return {
        "interactions": [
            {
                "moment_id": "moment-0",
                "interaction": copy.deepcopy(
                    draft["moments"][0]["interaction"]
                ),
            }
        ],
        "transfer_item": transfer_item,
    }


def test_director_contract_is_strictly_narrative_only():
    payload = json.loads(director_prompt(problem(), ["2", "3"], None, None, 2))
    schema = payload["output_contract"]["schema"]
    serialized = json.dumps(schema, ensure_ascii=False)

    assert payload["narrative_schema"] == NarrativeDraft.model_json_schema()
    assert "transfer_item" not in schema["properties"]
    assert "interaction" not in schema["$defs"]["NarrativeMoment"][
        "properties"
    ]
    assert "interaction_intent" in schema["$defs"]["NarrativeMoment"][
        "properties"
    ]
    assert "不得包含 interaction 字段" in DIRECTOR_SYSTEM
    assert "不得输出 transfer_item" in DIRECTOR_SYSTEM


def test_materials_contract_is_small_and_receives_validated_narrative():
    narrative = NarrativeDraft.model_validate(narrative_payload())
    payload = json.loads(
        materials_prompt(
            problem(),
            narrative,
            ["2", "3"],
            original_equation_degree=2,
        )
    )

    assert payload["validated_narrative"]["title"] == narrative.title
    assert payload["output_contract"]["schema"] == (
        MaterialsDraft.model_json_schema()
    )
    serialized_schema = json.dumps(
        payload["output_contract"]["schema"],
        ensure_ascii=False,
    )
    assert "point_select" not in serialized_schema
    assert "feedback_audio_url" not in serialized_schema
    assert '"label"' not in json.dumps(
        payload["output_contract"]["schema"]["$defs"][
            "GeneratedTransferOption"
        ]
    )
    assert "lesson_schema" not in payload
    assert "expected_answer=option_id" in MATERIALS_SYSTEM
    assert "互动前" in MATERIALS_SYSTEM


def test_reviewer_contract_uses_frozen_resolved_method_display():
    assert "resolved_method.display_name" in REVIEWER_SYSTEM
    assert "名称与 required_method 不一致" not in REVIEWER_SYSTEM


def test_narrative_schema_bounds_tts_fields_lists_and_board_actions():
    schema = NarrativeDraft.model_json_schema()
    properties = schema["properties"]
    moment = schema["$defs"]["NarrativeMoment"]["properties"]
    board_action = schema["$defs"]["NarrativeBoardAction"]["properties"]
    route_schema = MathRouteDraft.model_json_schema()
    math_step = route_schema["$defs"]["NarrativeMathStep"]["properties"]

    assert properties["title"]["maxLength"] == 120
    assert properties["opening"]["maxLength"] == 90
    assert properties["summary"]["maxLength"] == 90
    assert properties["moments"]["maxItems"] == 16
    assert "math_steps" not in properties
    assert route_schema["properties"]["math_steps"]["maxItems"] == 16
    assert moment["purpose"]["maxLength"] == 120
    assert moment["board_actions"]["maxItems"] == 12
    assert math_step["state_before"]["maxItems"] == 4
    assert math_step["state_after"]["maxItems"] == 4
    assert board_action["content"]["anyOf"][0]["maxLength"] == 500


def oversized_narrative_payload():
    payload = narrative_payload()
    source = copy.deepcopy(payload["moments"][1])
    moments = []
    for index in range(16):
        moment = copy.deepcopy(source)
        moment["moment_id"] = f"large-moment-{index}"
        moment["interaction_intent"] = (
            "诊断一个关键判断。"
            if index == 0
            else None
        )
        moment["board_actions"] = [
            {
                "type": "write",
                "target": f"target-{action_index}",
                "content": "x" * 500,
            }
            for action_index in range(12)
        ]
        moments.append(moment)
    payload["moments"] = moments
    return payload


def test_aggregate_narrative_size_gate_retries_once_with_safe_error():
    oversized = oversized_narrative_payload()
    client = FakeClient([oversized, copy.deepcopy(oversized)])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="教学主线整体内容过长"):
        asyncio.run(service.generate(problem()))

    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        DIRECTOR_SYSTEM,
    ]
    retry_payload = json.loads(client.all_calls[1][1])
    assert retry_payload["previous_validation_error"] == (
        "教学主线整体内容过长。"
    )
    assert "x" * 100 not in json.dumps(
        retry_payload["output_contract"]["retry"],
        ensure_ascii=False,
    )


def test_compose_deep_copies_nested_narrative_board_actions():
    narrative = NarrativeDraft.model_validate(narrative_payload())
    materials = MaterialsDraft.model_validate(materials_payload())
    service = LessonGenerationService(FakeClient([]), MathEngine())
    before = narrative.model_dump()
    verified_route = _VerifiedMathRoute.freeze(
        MathRouteDraft.model_validate(
            {"math_steps": valid_draft()["math_steps"]}
        ),
        "factor",
    )

    draft = service._compose_draft(
        narrative,
        materials,
        verified_route,
    )
    draft.moments[0].board_actions[0].target = "mutated-target"

    assert narrative.model_dump() == before
    assert narrative.moments[0].board_actions[0].target != "mutated-target"


def test_service_composes_materials_before_reviewer_and_reviewer_sees_whole_draft():
    client = FakeClient(
        [narrative_payload(), materials_payload(), approved_review()]
    )
    service = LessonGenerationService(client, MathEngine())

    lesson = asyncio.run(service.generate(problem()))

    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    reviewer_payload = json.loads(client.all_calls[2][1])
    assert reviewer_payload["whole_lesson"]["moments"][0]["interaction"][
        "interaction_id"
    ] == "find-factor-pair"
    assert reviewer_payload["whole_lesson"]["moments"][0]["layer"] == "base"
    assert reviewer_payload["whole_lesson"]["transfer_item"][
        "correct_option_id"
    ] == "both-roots"
    assert lesson.transfer_item.options[0].label == (
        r"\(x=3\) 或 \(x=4\)"
    )


@pytest.mark.parametrize("count", [0, 4])
def test_materials_reject_zero_or_four_interactions_after_one_retry(count):
    invalid = materials_payload()
    invalid["interactions"] = [
        {
            "moment_id": f"moment-{index % 2}",
            "interaction": {
                **copy.deepcopy(
                    valid_draft()["moments"][0]["interaction"]
                ),
                "interaction_id": f"choice-{index}",
            },
        }
        for index in range(count)
    ]
    client = FakeClient([narrative_payload(), invalid, invalid])
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="互动素材结构无效"):
        asyncio.run(service.generate(problem()))

    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        MATERIALS_SYSTEM,
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["interactions"][0].update(
                moment_id="missing-moment"
            ),
            "绑定位置",
        ),
        (
            lambda payload: payload["interactions"].append(
                {
                    "moment_id": "moment-0",
                    "interaction": {
                        **copy.deepcopy(
                            valid_draft()["moments"][0]["interaction"]
                        ),
                        "interaction_id": "second-choice",
                    },
                }
            ),
            "重复绑定",
        ),
        (
            lambda payload: payload["interactions"][0][
                "interaction"
            ].update(interaction_id="near-transfer"),
            "保留值",
        ),
    ],
)
def test_invalid_material_bindings_are_retried_once_then_rejected(
    mutate,
    message,
):
    invalid = materials_payload()
    mutate(invalid)
    client = FakeClient(
        [narrative_payload(), invalid, copy.deepcopy(invalid)]
    )

    with pytest.raises(LessonQualityError, match=message):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(problem())
        )

    retry_payload = json.loads(client.all_calls[2][1])
    assert retry_payload["previous_validation_error"]


def test_materials_cannot_bind_a_moment_without_director_intent():
    invalid = materials_payload()
    invalid["interactions"][0]["moment_id"] = "moment-1"
    client = FakeClient(
        [narrative_payload(), invalid, copy.deepcopy(invalid)]
    )

    with pytest.raises(LessonQualityError, match="已声明的互动意图"):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(problem())
        )


def test_materials_must_fill_every_director_interaction_intent():
    narrative = narrative_payload()
    narrative["moments"][1]["interaction_intent"] = (
        "诊断学生能否把因式分解结果连接到零乘积性质。"
    )
    incomplete = materials_payload()
    client = FakeClient(
        [narrative, incomplete, copy.deepcopy(incomplete)]
    )

    with pytest.raises(LessonQualityError, match="完整填写"):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(problem())
        )


def test_director_duplicate_moment_ids_fail_before_materials():
    invalid = narrative_payload()
    invalid["moments"][1]["moment_id"] = invalid["moments"][0]["moment_id"]
    client = FakeClient([invalid, copy.deepcopy(invalid)])

    with pytest.raises(LessonQualityError, match="讲解结构无效"):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(problem())
        )

    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        DIRECTOR_SYSTEM,
    ]


def test_materials_math_error_is_not_silently_corrected():
    invalid = materials_payload()
    invalid["transfer_item"]["expected_answer"] = "x=30 或 x=40"
    invalid["transfer_item"]["options"][0]["canonical_answer"] = (
        "x=30 或 x=40"
    )
    client = FakeClient(
        [narrative_payload(), invalid, copy.deepcopy(invalid)]
    )
    service = LessonGenerationService(client, MathEngine())

    with pytest.raises(LessonQualityError, match="近迁移题"):
        asyncio.run(service.generate(problem()))

    assert client.responses == []


def test_materials_reject_overlong_tts_facing_feedback():
    invalid = materials_payload()
    invalid["interactions"][0]["interaction"]["options"][0]["feedback"] = (
        "太长" * 91
    )
    client = FakeClient(
        [narrative_payload(), invalid, copy.deepcopy(invalid)]
    )

    with pytest.raises(LessonQualityError, match="互动素材结构无效"):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(problem())
        )


def test_materials_provider_failure_is_retried_without_reusing_a_response():
    client = FakeClient(
        [
            narrative_payload(),
            ModelResponseError("temporary-provider-detail"),
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(problem())
    )

    assert lesson.validation_report["review_status"] == "approved"
    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    assert client.all_calls[1] == client.all_calls[2]


def test_revision_rebuilds_narrative_then_regenerates_all_materials():
    revised_narrative = narrative_payload()
    revised_narrative["opening"] = "先明确条件，再寻找满足条件的因数对。"
    client = FakeClient(
        [
            narrative_payload(),
            materials_payload(),
            revision_review(),
            revised_narrative,
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(problem())
    )

    assert lesson.validation_report["revision_count"] == 1
    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
        REVISION_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    revision_payload = json.loads(client.all_calls[3][1])
    assert "current_narrative" in revision_payload
    assert revision_payload["resolved_method"] == {
        "family": "factor",
        "display_name": "因式分解法",
    }
    assert "transfer_item" not in revision_payload["output_contract"]
    assert "moment_choice" not in revision_payload["output_contract"]
    regenerated_materials = json.loads(client.all_calls[4][1])
    assert regenerated_materials["review"]["must_fix"] == (
        revision_review()["must_fix"]
    )


def test_oversized_revision_retries_before_regenerating_materials():
    revised = narrative_payload()
    revised["opening"] = "先明确条件，再寻找满足条件的因数对。"
    client = FakeClient(
        [
            narrative_payload(),
            materials_payload(),
            revision_review(),
            oversized_narrative_payload(),
            revised,
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(problem())
    )

    assert lesson.validation_report["revision_count"] == 1
    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
        REVISION_SYSTEM,
        REVISION_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    retry_payload = json.loads(client.all_calls[4][1])
    assert retry_payload["previous_validation_error"] == (
        "教学主线整体内容过长。"
    )


def test_raw_reference_solution_is_only_sent_to_reference_auditor():
    raw_marker = "RAW_REFERENCE_PRIVATE_MARKER"
    source = problem(reference_solution_text=raw_marker)
    audit = approved_audit()
    audit["evidence"] = [raw_marker]
    client = FakeClient(
        [
            audit,
            narrative_payload(),
            materials_payload(),
            revision_review(),
            narrative_payload(),
            materials_payload(),
            approved_review(),
        ]
    )

    asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(source)
    )

    assert raw_marker in client.all_calls[0][1]
    assert all(
        raw_marker not in user_prompt
        for _, user_prompt in client.all_calls[1:]
    )
