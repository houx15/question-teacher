import json
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_pedagogy_evaluation as evaluation
from app.pedagogy_rubric import (
    NON_COMPENSABLE_CRITERIA,
    PEDAGOGY_RUBRIC_VERSION,
    rubric_payload,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "pedagogy_golden_cases.json"

REQUIRED_COVERAGE_TAGS = {
    "concept_condition_conversion",
    "algebra_execution",
    "equation_parameter",
    "method_selection",
    "text_only_geometry",
    "function_relationship",
    "omitted_condition",
    "exploration_or_revision",
    "concept_overlay",
    "no_forced_interaction",
    "no_forced_emphasis",
}
CASE_KEYS = {
    "case_id",
    "problem",
    "coverage_tags",
    "trace_anchors",
    "required_reasoning_modes",
    "required_must_teach",
    "typical_misconceptions",
    "required_board_states",
    "acceptable_excerpt_patterns",
    "unacceptable_excerpt_patterns",
}
STRUCTURED_EXPECTATION_KEYS = {
    "required_step_labels",
    "required_spoken_forms",
    "required_error_codes",
}
PROBLEM_KEYS = {
    "problem_text",
    "reference_answer",
    "reference_solution_text",
    "lesson_length",
}
REASONING_MODES = {
    "understand",
    "plan",
    "explore",
    "execute",
    "monitor",
    "revise",
    "reflect",
}
EXPECTED_NEW_CRITERIA = {
    "step_purpose_and_transition",
    "display_speech_math_alignment",
    "adaptive_explanation_depth",
    "continuous_board_structure",
    "current_step_visible",
}


def test_rubric_contains_structured_teaching_hard_gates():
    payload = rubric_payload()
    criteria = {
        item["criterion_id"]: item
        for item in payload["criteria"]
    }

    assert PEDAGOGY_RUBRIC_VERSION == "0.2"
    assert EXPECTED_NEW_CRITERIA <= set(criteria)
    assert all(
        criteria[criterion_id]["non_compensable"] is False
        for criterion_id in EXPECTED_NEW_CRITERIA
    )
    assert set(NON_COMPENSABLE_CRITERIA) == {
        "current_emphasis_correct",
        "learner_follows_why",
    }


def _load_cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_golden_fixture_contains_exactly_18_unique_reviewable_cases():
    cases = _load_cases()

    assert isinstance(cases, list)
    assert len(cases) == 18
    case_ids = [case["case_id"] for case in cases]
    assert len(case_ids) == len(set(case_ids))
    assert all(re.fullmatch(r"[a-z][a-z0-9_]{2,63}", item) for item in case_ids)

    for case in cases:
        assert frozenset(case) in {
            frozenset(CASE_KEYS),
            frozenset(CASE_KEYS | STRUCTURED_EXPECTATION_KEYS),
        }
        assert set(case["problem"]) == PROBLEM_KEYS
        assert case["problem"]["lesson_length"] in {"concise", "standard"}
        for key in ("problem_text", "reference_answer", "reference_solution_text"):
            value = case["problem"][key]
            assert isinstance(value, str)
            assert value.strip() == value
            assert 1 <= len(value) <= 1200
        for key in CASE_KEYS - {"case_id", "problem"}:
            values = case[key]
            assert isinstance(values, list)
            assert 1 <= len(values) <= 8
            assert len(values) == len(set(values))
            assert all(
                isinstance(value, str)
                and value.strip() == value
                and 1 <= len(value) <= 160
                for value in values
            )
        assert set(case["coverage_tags"]).issubset(REQUIRED_COVERAGE_TAGS)
        assert set(case["required_reasoning_modes"]).issubset(REASONING_MODES)


def test_golden_fixture_starts_with_approved_parameter_root_case():
    case = _load_cases()[0]
    assert {key: case[key] for key in CASE_KEYS} == {
        "case_id": "parameter_root_01",
        "problem": {
            "problem_text": "若2n（n不等于0）是方程x^2-2mx+2n=0的根，求m-n。",
            "reference_answer": "1/2",
            "reference_solution_text": "将x=2n代入，得到4n^2-4mn+2n=0；由n不等于0，整理得4n-4m+2=0，所以m-n=1/2。",
            "lesson_length": "standard",
        },
        "coverage_tags": ["equation_parameter", "omitted_condition"],
        "trace_anchors": ["是根意味着代入", "约去n前使用n不等于0", "结果重新连接m-n"],
        "required_reasoning_modes": ["plan", "execute", "monitor"],
        "required_must_teach": ["方程的根代入后等式仍成立", "关于x的方程只代入x", "计算(2n)^2时先平方整体", "含字母因子约分前确认非零", "n-m与m-n互为相反数"],
        "typical_misconceptions": ["把2n代给m或n", "把(2n)^2算成2n^2", "直接除以n却不检查非零条件", "把n-m直接当作m-n"],
        "required_board_states": ["方程的根→代入后等式成立", "x=2n代入原方程", "(2n)^2-4mn+2n=0", "因为n不等于0约去n", "-4(m-n)+2=0", "m-n=1/2"],
        "acceptable_excerpt_patterns": ["是关于x的方程", "平方的是整个2n", "因为n不等于0", "n-m与m-n互为相反数"],
        "unacceptable_excerpt_patterns": ["直接把n约掉", "把2n代入m", "(2n)^2=2n^2", "答案显然是二分之一"],
    }
    assert case["required_step_labels"] == [
        "第一步：理解方程的根",
        "第二步：代入方程",
        "第三步：展开并化简",
        "第四步：利用n不等于0约去n",
        "第五步：整理出m-n",
    ]
    assert case["required_spoken_forms"] == [
        {"display": "m-n", "spoken_contains": "m 减 n"},
        {"display": "n≠0", "spoken_contains": "n 不等于零"},
    ]
    assert case["required_error_codes"] == [
        "substitution-variable-error",
        "square-distribution-error",
        "nonzero-condition-error",
        "opposite-expression-error",
    ]


def test_golden_fixture_covers_reviewed_breadth_without_templated_expectations():
    cases = _load_cases()
    coverage = {
        tag for case in cases for tag in case["coverage_tags"]
    }
    assert coverage == REQUIRED_COVERAGE_TAGS

    normalized_problems = {
        re.sub(r"\s+", "", case["problem"]["problem_text"]).lower()
        for case in cases
    }
    assert len(normalized_problems) == 18

    anchors = [
        anchor
        for case in cases
        for anchor in case["trace_anchors"]
    ]
    assert len(anchors) == len(set(anchors))
    banned_slogans = {"认真审题", "仔细计算", "掌握方法", "注意易错点", "理解为什么"}
    assert not banned_slogans.intersection(anchors)

    families = {
        "方程": any("方程" in case["problem"]["problem_text"] for case in cases),
        "函数": any("函数" in case["problem"]["problem_text"] for case in cases),
        "几何": any(
            token in case["problem"]["problem_text"]
            for case in cases
            for token in ("三角形", "平行四边形", "圆")
        ),
        "统计概率": any(
            token in case["problem"]["problem_text"]
            for case in cases
            for token in ("平均数", "概率")
        ),
        "不等式": any("不等式" in case["problem"]["problem_text"] for case in cases),
    }
    assert all(families.values())


def _structured_evidence_record(case):
    clauses = [
        SimpleNamespace(
            clause_id="clause-%d" % index,
            display_text=content,
            spoken_text=content,
            must_teach_refs=["must-%d" % index],
        )
        for index, content in enumerate(case["required_must_teach"], start=1)
    ]
    clauses[3].display_text += " n≠0"
    clauses[3].spoken_text += " n 不等于零"
    clauses[4].display_text += " m-n"
    clauses[4].spoken_text = "n 减 m 与 m 减 n 互为相反数"
    items = [
        SimpleNamespace(
            must_teach_id="must-%d" % index,
            content=content,
            student_display_evidence=content,
            student_spoken_evidence=clauses[index - 1].spoken_text,
        )
        for index, content in enumerate(case["required_must_teach"], start=1)
    ]
    options = [
        SimpleNamespace(error_code=error_code)
        for error_code in case["required_error_codes"]
    ]
    prepared = SimpleNamespace(
        reasoning_trajectory=SimpleNamespace(
            episodes=[SimpleNamespace(must_teach=items)]
        ),
        teaching_progression=SimpleNamespace(
            steps=[
                SimpleNamespace(
                    step_id="step-%d" % index,
                    directory_label=label,
                )
                for index, label in enumerate(case["required_step_labels"], start=1)
            ]
        ),
        teaching_script=SimpleNamespace(
            clauses=clauses,
            response_scripts=[
                SimpleNamespace(
                    classification="correct",
                    depth="brief",
                    clauses=[SimpleNamespace(must_teach_refs=[])],
                ),
                SimpleNamespace(
                    classification="incorrect",
                    depth="worked",
                    clauses=[SimpleNamespace(must_teach_refs=[])],
                ),
            ],
        ),
        interaction_plan=SimpleNamespace(
            interactions=[SimpleNamespace(options=options)]
        ),
        performance_score=SimpleNamespace(
            cues=[
                SimpleNamespace(
                    lead_actions=[],
                    start_actions=[
                        SimpleNamespace(
                            clause_id=clause.clause_id,
                            action=SimpleNamespace(
                                type="write",
                                surface="board",
                                content=items[index - 1].student_display_evidence,
                                teaching_step_id="step-%d" % index,
                            ),
                        )
                    ],
                    end_actions=[
                        SimpleNamespace(
                            clause_id=clause.clause_id,
                            action=SimpleNamespace(
                                type="complete_step",
                                surface="board",
                                content=None,
                                teaching_step_id="step-%d" % index,
                            ),
                        )
                    ],
                )
                for index, clause in enumerate(clauses, start=1)
            ]
        ),
    )
    return SimpleNamespace(prepared_lesson=prepared)


@pytest.mark.parametrize("mutation", ("label", "spoken", "error_code"))
def test_parameter_root_structured_evidence_fails_closed(mutation):
    case = _load_cases()[0]
    record = _structured_evidence_record(case)
    if mutation == "label":
        record.prepared_lesson.teaching_progression.steps[0].directory_label = (
            "第一步：缺少根的意义"
        )
    elif mutation == "spoken":
        record.prepared_lesson.teaching_script.clauses[4].spoken_text = "m n"
    else:
        record.prepared_lesson.interaction_plan.interactions[0].options[
            0
        ].error_code = "other-error"

    with pytest.raises(evaluation.GoldenEvidenceError):
        evaluation._validate_structured_case_evidence(case, record)


@pytest.mark.parametrize(
    "mutation",
    (
        "must_teach_1",
        "must_teach_2",
        "must_teach_3",
        "must_teach_4",
        "must_teach_5",
        "minus_not_spoken",
        "wrong_support_not_deeper",
        "step_never_completed",
    ),
)
def test_parameter_root_eight_required_mutations_fail_closed(mutation):
    case = _load_cases()[0]
    record = _structured_evidence_record(case)
    prepared = record.prepared_lesson
    if mutation.startswith("must_teach_"):
        index = int(mutation.rsplit("_", 1)[1]) - 1
        prepared.teaching_script.clauses[index].must_teach_refs = []
    elif mutation == "minus_not_spoken":
        prepared.teaching_script.clauses[4].spoken_text = "n m 互为相反数"
    elif mutation == "wrong_support_not_deeper":
        prepared.teaching_script.response_scripts[1].depth = "brief"
    else:
        prepared.performance_score.cues[2].end_actions = []

    with pytest.raises(evaluation.GoldenEvidenceError):
        evaluation._validate_structured_case_evidence(case, record)


def _fake_bundle(case_id="case_one"):
    clauses = [
        SimpleNamespace(clause_id="clause-1", must_teach_refs=["teach-1"]),
        SimpleNamespace(clause_id="clause-2", must_teach_refs=["teach-2"]),
    ]
    episodes = [
        SimpleNamespace(
            must_teach=[SimpleNamespace(must_teach_id="teach-1")]
        ),
        SimpleNamespace(
            must_teach=[SimpleNamespace(must_teach_id="teach-2")]
        ),
    ]
    actions = [SimpleNamespace(clause_id="clause-1")]
    prepared = SimpleNamespace(
        rubric_version="0.1",
        review=SimpleNamespace(status="approved"),
        reasoning_trajectory=SimpleNamespace(episodes=episodes),
        teaching_script=SimpleNamespace(clauses=clauses),
        performance_score=SimpleNamespace(
            cues=[
                SimpleNamespace(
                    clause_ids=["clause-1"],
                    lead_actions=[],
                    start_actions=actions,
                    end_actions=[],
                ),
                SimpleNamespace(
                    clause_ids=["clause-2"],
                    lead_actions=[],
                    start_actions=[],
                    end_actions=[],
                ),
            ]
        ),
    )
    record = SimpleNamespace(
        prepared_lesson=prepared,
        role_calls=[object()] * 7,
        model_dump=lambda mode="json": {
            "generation_id": "generation-%s" % case_id,
            "lesson_id": "lesson-%s" % case_id,
            "prepared_lesson": {"rubric_version": "0.1"},
            "role_calls": [{"role": "redacted-private"}] * 7,
        },
    )
    lesson = SimpleNamespace(
        lesson_id="lesson-%s" % case_id,
        model_dump=lambda mode="json": {
            "lesson_id": "lesson-%s" % case_id,
            "problem": {
                "problem_text": "公开题目",
                "reference_answer": "秘密答案",
                "reference_solution_text": "秘密解析",
                "required_method": None,
                "lesson_length": "concise",
            },
            "title": "公开讲解",
            "learning_goal": "理解原因",
            "beats": [{
                "beat_id": "beat-1",
                "purpose": "解释",
                "narration": "两边同时减一。",
                "board_actions": [],
                "layer": "base",
                "sync_cues": [{
                    "cue_id": "cue-1",
                    "spoken_text": "两边同时减一。",
                    "lead_actions": [],
                    "start_actions": [],
                    "end_actions": [],
                    "audio_url": None,
                    "private_feedback": "nested secret",
                }],
                "interaction": None,
                "audio_url": None,
                "next_beat_id": None,
                "validation_report": "nested secret",
            }],
            "summary": "公开总结",
            "transfer_item": {
                "problem_text": "迁移题",
                "expected_answer": "秘密",
                "method_signal": "同一方法",
                "options": [],
                "correct_option_id": None,
            },
            "validation_report": {"private_feedback": "provider secret"},
            "candidate_version": "prompt-secret",
        },
    )
    return SimpleNamespace(lesson=lesson, generation_record=record)


class _FakeService:
    async def generate_bundle(self, problem, on_stage=None):
        if on_stage is not None:
            await on_stage("正在设计解题思维轨迹")
        return _fake_bundle(problem.problem_text)


def _single_case():
    return [{
        "case_id": "case_one",
        "problem": {
            "problem_text": "x+1=2",
            "reference_answer": "x=1",
            "reference_solution_text": "两边减1得x=1。",
            "lesson_length": "concise",
        },
        "coverage_tags": ["algebra_execution"],
        "trace_anchors": ["两边同步减一"],
        "required_reasoning_modes": ["execute"],
        "required_must_teach": ["等式两边同步变形"],
        "typical_misconceptions": ["只在左边减一"],
        "required_board_states": ["x+1=2到x=1"],
        "acceptable_excerpt_patterns": ["两边同时减一"],
        "unacceptable_excerpt_patterns": ["直接移过去"],
    }]


def test_contract_metric_accepts_action_bound_to_response_clause():
    bundle = _fake_bundle()
    prepared = bundle.generation_record.prepared_lesson
    response_clause = SimpleNamespace(
        clause_id="response-clause-1",
        must_teach_refs=[],
        display_text="错误原因",
        spoken_text="我们拆解这个错误原因。",
        lesson_step_id=None,
    )
    prepared.teaching_script.response_scripts = [
        SimpleNamespace(
            interaction_id="interaction-1",
            option_id="option-wrong",
            classification="incorrect",
            depth="worked",
            clauses=[response_clause],
        )
    ]
    prepared.interaction_plan = SimpleNamespace(interactions=[])
    prepared.performance_score.cues.append(
        SimpleNamespace(
            clause_ids=[response_clause.clause_id],
            lead_actions=[],
            start_actions=[
                SimpleNamespace(
                    clause_id=response_clause.clause_id,
                    action=SimpleNamespace(
                        surface="board",
                        type="write",
                        teaching_step_id=None,
                    ),
                )
            ],
            end_actions=[],
        )
    )

    metrics = evaluation._contract_metrics(
        bundle.lesson,
        bundle.generation_record,
        duration_ms=1,
        rubric_version="0.1",
        call_count=7,
    )

    assert metrics["clause_action_binding"] == {
        "valid": 2,
        "total": 2,
        "ratio": 1.0,
    }


def _canonical_sha256(payload):
    encoded = evaluation._json_content(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _public_artifact(title):
    return {
        "schema_version": 1,
        "lesson_id": "lesson-case_one",
        "problem": {
            "problem_text": "公开题目",
            "required_method": None,
            "lesson_length": "concise",
        },
        "title": title,
        "learning_goal": "理解原因",
        "beats": [{
            "beat_id": "beat-1",
            "purpose": "解释",
            "narration": "两边同时减一。",
            "layer": "base",
            "sync_cues": [{
                "cue_id": "cue-1",
                "spoken_text": "两边同时减一。",
                "lead_actions": [],
                "start_actions": [],
                "end_actions": [],
            }],
            "interaction": None,
            "next_beat_id": None,
        }],
        "summary": "公开总结",
        "transfer_item": {
            "problem_text": "迁移题",
            "method_signal": "同一方法",
            "options": [],
        },
    }


def _success_metrics():
    return {
        "generation_success": True,
        "hard_gate_review_pass": True,
        "must_teach_coverage": {"covered": 1, "total": 1, "ratio": 1.0},
        "step_coverage": {"covered": 0, "total": 0, "ratio": 1.0},
        "must_teach_to_script_coverage": {"covered": 1, "total": 1, "ratio": 1.0},
        "must_teach_to_board_coverage": {"covered": 0, "total": 1, "ratio": 0.0},
        "display_speech_alignment": {"aligned": 0, "total": 0, "ratio": 1.0},
        "diagnostic_branch_coverage": {"covered": 0, "total": 0, "ratio": 1.0},
        "step_lifecycle_coverage": {"complete": 0, "total": 0, "ratio": 1.0},
        "clause_action_binding": {"valid": 0, "total": 0, "ratio": 1.0},
        "schema_runtime_pass": True,
        "duration_ms": 10,
        "call_count": 7,
    }


def _failure_metrics():
    return {
        "generation_success": False,
        "hard_gate_review_pass": False,
        "must_teach_coverage": None,
        "step_coverage": None,
        "must_teach_to_script_coverage": None,
        "must_teach_to_board_coverage": None,
        "display_speech_alignment": None,
        "diagnostic_branch_coverage": None,
        "step_lifecycle_coverage": None,
        "clause_action_binding": None,
        "schema_runtime_pass": False,
        "duration_ms": 10,
        "call_count": 3,
    }


def _write_run_directory(directory, candidate, statuses):
    runtime = directory / "public" / "runtime"
    runtime.mkdir(parents=True)
    runs = []
    for run_index, status in enumerate(statuses, start=1):
        identity = {
            "case_id": "case_one",
            "run_index": run_index,
        }
        if status == "succeeded":
            artifact = _public_artifact("公开讲解%d" % run_index)
            (runtime / ("case_one__run-%02d.json" % run_index)).write_text(
                evaluation._json_content(artifact), encoding="utf-8"
            )
            runs.append({
                **identity,
                "status": "succeeded",
                "public_sha256": _canonical_sha256(artifact),
                "metrics": _success_metrics(),
            })
        else:
            runs.append({
                **identity,
                "status": "failed",
                "failure": {
                    "category": "provider_error",
                    "stage": "script",
                },
                "metrics": _failure_metrics(),
            })
    manifest = {
        "schema_version": 1,
        "rubric_version": "0.1",
        "candidate_version": candidate,
        "runs_per_case": len(statuses),
        "case_ids": ["case_one"],
        "case_set_sha256": "a" * 64,
        "metric_scope": list(_success_metrics()),
        "evidence_boundary": "contracts only",
        "runs": runs,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _walk_keys(value):
    if type(value) is dict:
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif type(value) is list:
        for item in value:
            yield from _walk_keys(item)


def test_generation_cli_requires_explicit_contract_arguments():
    parser = evaluation.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args([
        "--rubric-version", "0.1",
        "--runs-per-case", "3",
        "--output-dir", "/tmp/evaluation-output",
        "--candidate-version", "prompt-2026-08-12",
    ])
    assert args.rubric_version == "0.1"
    assert args.runs_per_case == 3
    assert args.candidate_version == "prompt-2026-08-12"


def test_runtime_fixture_validation_rejects_path_ids_and_duplicates(tmp_path):
    case = _single_case()[0]
    for case_id in ("../escape", "bad/id"):
        path = tmp_path / (case_id.replace("/", "_") + ".json")
        path.write_text(json.dumps([{**case, "case_id": case_id}]), encoding="utf-8")
        with pytest.raises(evaluation.EvaluationConfigurationError, match="case"):
            evaluation._load_cases(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps([case, case]), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationConfigurationError, match="unique"):
        evaluation._load_cases(duplicate)


def test_counting_client_counts_route_and_preparation_call_styles():
    class Delegate:
        async def complete_json(self, system, user):
            del system, user
            return {"route": True}

        async def complete_json_with_metadata(self, system, user):
            del system, user
            return {"prepared": True}

    client = evaluation.CountingModelClient(Delegate())

    async def exercise():
        await client.complete_json("route", "input")
        await client.complete_json_with_metadata("role", "input")

    import asyncio
    asyncio.run(exercise())
    assert client.call_count == 2


def test_runner_refuses_offline_execution_and_nonempty_output(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_INTEGRATION", raising=False)
    with pytest.raises(evaluation.EvaluationConfigurationError, match="RUN_INTEGRATION"):
        evaluation.run_evaluation(
            _single_case(),
            rubric_version="0.1",
            runs_per_case=1,
            output_dir=tmp_path / "offline",
            service_factory=lambda: _FakeService(),
        )

    monkeypatch.setenv("RUN_INTEGRATION", "1")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(evaluation.EvaluationConfigurationError, match="non-empty"):
        evaluation.run_evaluation(
            _single_case(),
            rubric_version="0.1",
            runs_per_case=1,
            output_dir=occupied,
            service_factory=lambda: _FakeService(),
        )


def test_runner_separates_private_records_and_public_summaries(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_INTEGRATION", "1")
    output = tmp_path / "run"
    evaluation.run_evaluation(
        _single_case(),
        rubric_version="0.1",
        runs_per_case=1,
        output_dir=output,
        service_factory=lambda: _FakeService(),
        candidate_version="prompt-a",
        clock=evaluation.StepClock([10.0, 10.125]),
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    metric = manifest["runs"][0]["metrics"]
    assert metric == {
        "generation_success": True,
        "hard_gate_review_pass": True,
        "must_teach_coverage": {"covered": 2, "total": 2, "ratio": 1.0},
        "step_coverage": {"covered": 0, "total": 0, "ratio": 1.0},
        "must_teach_to_script_coverage": {"covered": 2, "total": 2, "ratio": 1.0},
        "must_teach_to_board_coverage": {"covered": 0, "total": 2, "ratio": 0.0},
        "display_speech_alignment": {"aligned": 0, "total": 0, "ratio": 1.0},
        "diagnostic_branch_coverage": {"covered": 0, "total": 0, "ratio": 1.0},
        "step_lifecycle_coverage": {"complete": 0, "total": 0, "ratio": 1.0},
        "clause_action_binding": {"valid": 1, "total": 1, "ratio": 1.0},
        "schema_runtime_pass": True,
        "duration_ms": 125,
        "call_count": 7,
    }
    assert manifest["candidate_version"] == "prompt-a"
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["runs"][0]["public_sha256"])
    private_text = (output / "private" / "records" / "case_one__run-01.json").read_text(encoding="utf-8")
    public_text = (output / "public" / "runtime" / "case_one__run-01.json").read_text(encoding="utf-8")
    public_payload = json.loads(public_text)
    assert "redacted-private" in private_text
    assert set(_walk_keys(public_payload)).isdisjoint({
        "candidate_version",
        "validation_report",
        "private_feedback",
        "reference_answer",
        "reference_solution_text",
        "expected_answer",
        "correct_option_id",
        "canonical_answer",
        "feedback",
    })
    assert manifest["runs"][0]["public_sha256"] == _canonical_sha256(
        public_payload
    )
    assert manifest["runs"][0]["public_sha256"] == hashlib.sha256(
        (output / "public" / "runtime" / "case_one__run-01.json").read_bytes()
    ).hexdigest()
    assert "provider secret" not in (output / "run.log").read_text(encoding="utf-8")


def test_failed_runs_store_only_content_free_category_and_stage(tmp_path, monkeypatch):
    class ProviderFailure(RuntimeError):
        category = "provider_error"

    class FailingService:
        async def generate_bundle(self, problem, on_stage=None):
            del problem
            if on_stage is not None:
                await on_stage("正在编写讲稿")
            raise ProviderFailure("OPENAI_API_KEY=leak provider body")

    monkeypatch.setenv("RUN_INTEGRATION", "1")
    output = tmp_path / "failure"
    evaluation.run_evaluation(
        _single_case(),
        rubric_version="0.1",
        runs_per_case=1,
        output_dir=output,
        service_factory=lambda: FailingService(),
    )
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in manifest_text
    run = json.loads(manifest_text)["runs"][0]
    assert run["failure"] == {
        "category": "provider_error",
        "stage": "script",
    }
    assert run["metrics"]["generation_success"] is False


def test_blind_comparison_hides_candidate_versions_and_keeps_private_mapping(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run_directory(left, "prompt-a", ["succeeded"])
    _write_run_directory(right, "prompt-b", ["succeeded"])

    output = tmp_path / "comparison"
    evaluation.create_blind_comparison(left, right, output)
    pair_text = (output / "public" / "blind_pairs.json").read_text(encoding="utf-8")
    pairs = json.loads(pair_text)
    assert pairs[0]["pair_id"] == "case_one__run-01"
    assert set(pairs[0]) == {"pair_id", "case_id", "run_index", "candidate_a", "candidate_b"}
    assert "prompt-a" not in pair_text and "prompt-b" not in pair_text
    mapping_text = (output / "private" / "candidate_mapping.json").read_text(encoding="utf-8")
    assert "prompt-a" in mapping_text and "prompt-b" in mapping_text
    assert "teacher_preference" not in pair_text
    assert "learning_effect" not in pair_text
    assert set(_walk_keys(pairs)).isdisjoint({
        "candidate_version",
        "validation_report",
        "private_feedback",
        "reference_answer",
        "reference_solution_text",
        "expected_answer",
        "correct_option_id",
        "canonical_answer",
        "feedback",
    })


def test_blind_comparison_rejects_untrusted_manifest_run_identity(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for directory, candidate in ((left, "candidate-a"), (right, "candidate-b")):
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "rubric_version": "0.1",
            "candidate_version": candidate,
            "runs_per_case": 1,
            "case_ids": ["../escape"],
            "case_set_sha256": "a" * 64,
            "runs": [{
                "case_id": "../escape",
                "run_index": 1,
                "status": "succeeded",
            }],
        }), encoding="utf-8")

    with pytest.raises(evaluation.EvaluationConfigurationError, match="manifest"):
        evaluation.create_blind_comparison(
            left,
            right,
            tmp_path / "comparison",
        )


def test_comparison_verifies_public_artifact_hash_and_strict_schema(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run_directory(left, "candidate-a", ["succeeded"])
    _write_run_directory(right, "candidate-b", ["succeeded"])

    artifact_path = left / "public" / "runtime" / "case_one__run-01.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["validation_report"] = {"private_feedback": "leak"}
    artifact_path.write_text(
        evaluation._json_content(artifact), encoding="utf-8"
    )

    with pytest.raises(evaluation.EvaluationConfigurationError, match="hash"):
        evaluation.create_blind_comparison(
            left,
            right,
            tmp_path / "comparison-hash",
        )

    left_clean = tmp_path / "left-clean"
    _write_run_directory(left_clean, "candidate-c", ["succeeded"])
    clean_path = left_clean / "public" / "runtime" / "case_one__run-01.json"
    malformed = json.loads(clean_path.read_text(encoding="utf-8"))
    malformed["beats"][0]["sync_cues"][0]["private_feedback"] = "leak"
    clean_path.write_text(
        evaluation._json_content(malformed), encoding="utf-8"
    )
    left_manifest = json.loads(
        (left_clean / "manifest.json").read_text(encoding="utf-8")
    )
    left_manifest["runs"][0]["public_sha256"] = _canonical_sha256(malformed)
    (left_clean / "manifest.json").write_text(
        json.dumps(left_manifest), encoding="utf-8"
    )
    with pytest.raises(evaluation.EvaluationConfigurationError, match="schema"):
        evaluation.create_blind_comparison(
            left_clean,
            right,
            tmp_path / "comparison-schema",
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_manifest_requires_complete_cartesian_run_identity_set(
    tmp_path,
    mutation,
):
    run_dir = tmp_path / mutation
    manifest = _write_run_directory(
        run_dir,
        "candidate-a",
        ["succeeded", "failed"],
    )
    if mutation == "missing":
        manifest["runs"] = manifest["runs"][:1]
    elif mutation == "duplicate":
        manifest["runs"] = [manifest["runs"][0], manifest["runs"][0]]
    else:
        manifest["runs"].append({**manifest["runs"][0], "run_index": 3})
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(evaluation.EvaluationConfigurationError, match="complete"):
        evaluation._read_manifest(run_dir)


def test_comparison_reports_all_run_outcomes_and_pairs_only_both_success(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run_directory(
        left,
        "candidate-a",
        ["succeeded", "failed", "failed"],
    )
    _write_run_directory(
        right,
        "candidate-b",
        ["succeeded", "succeeded", "failed"],
    )

    output = tmp_path / "comparison"
    evaluation.create_blind_comparison(left, right, output)
    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    pairs = json.loads(
        (output / "public" / "blind_pairs.json").read_text(encoding="utf-8")
    )

    assert manifest["comparison_counts"] == {
        "matched_runs": 3,
        "left_successes": 1,
        "left_failures": 2,
        "right_successes": 2,
        "right_failures": 1,
        "both_success": 1,
        "both_failed": 1,
        "one_sided_failure": 1,
        "excluded_from_blind_pairs": 2,
        "blind_pairs": 1,
    }
    assert [pair["run_index"] for pair in pairs] == [1]


def test_output_guard_rejects_root_and_controlled_child_symlinks(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(evaluation.EvaluationConfigurationError, match="symlink"):
        evaluation._OutputGuard.create(root_link)

    root = tmp_path / "root"
    guard = evaluation._OutputGuard.create(root)
    guard.ensure_directory("public")
    (root / "public").rmdir()
    (root / "public").symlink_to(target, target_is_directory=True)
    with pytest.raises(evaluation.EvaluationConfigurationError, match="symlink"):
        guard.write_json("public/runtime.json", {"safe": True})


def test_output_guard_rejects_group_or_world_writable_existing_root(tmp_path):
    for mode in (0o720, 0o702, 0o777):
        root = tmp_path / ("root-%o" % mode)
        root.mkdir()
        root.chmod(mode)

        with pytest.raises(evaluation.EvaluationConfigurationError, match="permissions"):
            evaluation._OutputGuard.create(root)

        assert list(root.iterdir()) == []


@pytest.mark.parametrize("mode", [0o720, 0o702, 0o777])
def test_output_guard_rejects_replaced_writable_child_before_content_write(
    tmp_path,
    mode,
):
    root = tmp_path / "root"
    guard = evaluation._OutputGuard.create(root)
    child = guard.ensure_directory("public")
    child.chmod(mode)

    with pytest.raises(evaluation.EvaluationConfigurationError, match="permissions"):
        guard.write_json("public/runtime.json", {"safe": True})

    assert not (child / "runtime.json").exists()


def test_output_guard_forces_secure_modes_despite_restrictive_umask(tmp_path):
    root = tmp_path / "root"
    previous_umask = evaluation.os.umask(0o777)
    try:
        guard = evaluation._OutputGuard.create(root)
        child = guard.ensure_directory("public/runtime")
        guard.write_json("public/runtime/lesson.json", {"safe": True})
    finally:
        evaluation.os.umask(previous_umask)

    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "public").stat().st_mode & 0o777 == 0o700
    assert child.stat().st_mode & 0o777 == 0o700
    assert (child / "lesson.json").stat().st_mode & 0o777 == 0o600


def test_output_guard_leaf_write_cannot_follow_directory_swap(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "root"
    guard = evaluation._OutputGuard.create(root)
    guard.ensure_directory("public")
    original_open = Path.open
    raced = {"value": False}

    def swap_before_leaf_open(path, *args, **kwargs):
        if path == root / "public" / "runtime.json":
            (root / "public").rename(root / "held-public")
            (root / "public").symlink_to(external, target_is_directory=True)
            raced["value"] = True
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_before_leaf_open)
    try:
        guard.write_json("public/runtime.json", {"safe": True})
    except evaluation.EvaluationConfigurationError:
        pass

    assert raced["value"] is False
    assert not (external / "runtime.json").exists()


def test_output_guard_fails_closed_if_child_is_swapped_during_dirfd_walk(
    tmp_path,
    monkeypatch,
):
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "root"
    guard = evaluation._OutputGuard.create(root)
    guard.ensure_directory("public")
    real_open = evaluation.os.open
    raced = {"value": False}

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "public" and dir_fd is not None and not raced["value"]:
            (root / "public").rename(root / "held-public")
            (root / "public").symlink_to(external, target_is_directory=True)
            raced["value"] = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(evaluation.os, "open", racing_open)
    with pytest.raises(evaluation.EvaluationConfigurationError):
        guard.write_json("public/runtime.json", {"safe": True})

    assert raced["value"] is True
    assert not (external / "runtime.json").exists()


@pytest.mark.parametrize("failing_fstat_call", [1, 2])
def test_output_guard_closes_every_descriptor_when_validation_fails(
    tmp_path,
    monkeypatch,
    failing_fstat_call,
):
    guard = evaluation._OutputGuard.create(tmp_path / "root")
    real_open = evaluation.os.open
    real_fstat = evaluation.os.fstat
    opened = []
    calls = {"value": 0}

    def tracking_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def failing_fstat(descriptor):
        calls["value"] += 1
        if calls["value"] == failing_fstat_call:
            raise OSError("simulated descriptor validation failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(evaluation.os, "open", tracking_open)
    monkeypatch.setattr(evaluation.os, "fstat", failing_fstat)
    with pytest.raises(evaluation.EvaluationConfigurationError):
        guard.ensure_directory("public")

    assert opened
    for descriptor in opened:
        with pytest.raises(OSError):
            real_fstat(descriptor)


@pytest.mark.parametrize("wrong_owner_call", [1, 2])
def test_output_guard_rejects_wrong_owner_and_closes_descriptors(
    tmp_path,
    monkeypatch,
    wrong_owner_call,
):
    root = tmp_path / "root"
    guard = evaluation._OutputGuard.create(root)
    guard.ensure_directory("public")
    real_open = evaluation.os.open
    real_fstat = evaluation.os.fstat
    opened = []
    calls = {"value": 0}

    def tracking_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened.append(descriptor)
        return descriptor

    def wrong_owner_fstat(descriptor):
        result = real_fstat(descriptor)
        calls["value"] += 1
        if calls["value"] != wrong_owner_call:
            return result
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_dev=result.st_dev,
            st_ino=result.st_ino,
            st_uid=evaluation.os.geteuid() + 1,
            st_nlink=result.st_nlink,
        )

    monkeypatch.setattr(evaluation.os, "open", tracking_open)
    monkeypatch.setattr(evaluation.os, "fstat", wrong_owner_fstat)
    with pytest.raises(evaluation.EvaluationConfigurationError, match="owner"):
        guard.write_json("public/runtime.json", {"safe": True})

    assert not (root / "public" / "runtime.json").exists()
    assert opened
    for descriptor in opened:
        with pytest.raises(OSError):
            real_fstat(descriptor)


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("succeeded", "hard_gate_review_pass", False),
        ("succeeded", "schema_runtime_pass", False),
        ("failed", "hard_gate_review_pass", True),
        ("failed", "schema_runtime_pass", True),
        ("succeeded", "duration_ms", True),
        ("succeeded", "call_count", True),
    ],
)
def test_manifest_metrics_reject_status_and_boolean_contradictions(
    status,
    field,
    value,
):
    metrics = _success_metrics() if status == "succeeded" else _failure_metrics()
    metrics[field] = value

    with pytest.raises(evaluation.EvaluationConfigurationError, match="metrics"):
        evaluation._validate_manifest_metrics(metrics, status)


@pytest.mark.parametrize(
    ("field", "metric"),
    [
        (
            "must_teach_coverage",
            {"covered": True, "total": 1, "ratio": 1.0},
        ),
        (
            "must_teach_coverage",
            {"covered": 1, "total": 1, "ratio": 0.5},
        ),
        (
            "must_teach_coverage",
            {"covered": 1, "total": 1, "ratio": 1},
        ),
        (
            "must_teach_coverage",
            {"covered": 1, "total": 1, "ratio": float("nan")},
        ),
        (
            "clause_action_binding",
            {"valid": 1, "total": 2, "ratio": 0.75},
        ),
        (
            "clause_action_binding",
            {"valid": 0, "total": 0, "ratio": 0.0},
        ),
    ],
)
def test_manifest_metrics_require_exact_finite_count_ratios(field, metric):
    metrics = _success_metrics()
    metrics[field] = metric

    with pytest.raises(evaluation.EvaluationConfigurationError, match="metrics"):
        evaluation._validate_manifest_metrics(metrics, "succeeded")


def test_unhashable_case_and_manifest_values_fail_as_configuration_errors(tmp_path):
    malformed_case = _single_case()[0].copy()
    malformed_case["coverage_tags"] = [["unhashable"]]
    with pytest.raises(evaluation.EvaluationConfigurationError):
        evaluation._validate_cases([malformed_case])

    run_dir = tmp_path / "run"
    manifest = _write_run_directory(run_dir, "candidate-a", ["succeeded"])
    manifest["case_ids"] = [["unhashable"]]
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(evaluation.EvaluationConfigurationError):
        evaluation._read_manifest(run_dir)

    run_dir_two = tmp_path / "run-two"
    manifest_two = _write_run_directory(
        run_dir_two,
        "candidate-b",
        ["succeeded"],
    )
    manifest_two["runs"][0]["case_id"] = ["unhashable"]
    (run_dir_two / "manifest.json").write_text(
        json.dumps(manifest_two), encoding="utf-8"
    )
    with pytest.raises(evaluation.EvaluationConfigurationError):
        evaluation._read_manifest(run_dir_two)
