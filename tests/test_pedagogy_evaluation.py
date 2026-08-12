import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_pedagogy_evaluation as evaluation


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
        assert set(case) == CASE_KEYS
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
    assert _load_cases()[0] == {
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
        "required_must_teach": ["目标只需要m与n的关系", "含字母因子约分前确认非零"],
        "typical_misconceptions": ["直接除以n却不检查非零条件", "只算代入而不解释为什么"],
        "required_board_states": ["x=2n代入原方程", "提取2n后的关系式", "m-n=1/2"],
        "acceptable_excerpt_patterns": ["先看目标", "因为n不等于0"],
        "unacceptable_excerpt_patterns": ["直接把n约掉", "答案显然是二分之一"],
    }


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
            "problem": {"problem_text": "公开题目", "reference_answer": "秘密答案"},
            "title": "公开讲解",
            "learning_goal": "理解原因",
            "beats": [],
            "summary": "公开总结",
            "transfer_item": {
                "problem_text": "迁移题",
                "expected_answer": "秘密",
                "method_signal": "同一方法",
                "options": [],
                "correct_option_id": None,
            },
            "validation_report": {"private_feedback": "provider secret"},
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
        "clause_action_binding": {"valid": 1, "total": 1, "ratio": 1.0},
        "schema_runtime_pass": True,
        "duration_ms": 125,
        "call_count": 7,
    }
    assert manifest["candidate_version"] == "prompt-a"
    private_text = (output / "private" / "records" / "case_one__run-01.json").read_text(encoding="utf-8")
    public_text = (output / "public" / "runtime" / "case_one__run-01.json").read_text(encoding="utf-8")
    assert "redacted-private" in private_text
    assert "reference_answer" not in public_text
    assert "validation_report" not in public_text
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
    for directory, version, title in (
        (left, "0.1", "旧版公开讲解"),
        (right, "0.2", "新版公开讲解"),
    ):
        (directory / "public" / "runtime").mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "rubric_version": version,
            "candidate_version": "prompt-" + version,
            "runs_per_case": 1,
            "case_ids": ["case_one"],
            "case_set_sha256": "a" * 64,
            "runs": [{"case_id": "case_one", "run_index": 1, "status": "succeeded"}],
        }), encoding="utf-8")
        (directory / "public" / "runtime" / "case_one__run-01.json").write_text(
            json.dumps({"title": title}), encoding="utf-8"
        )

    output = tmp_path / "comparison"
    evaluation.create_blind_comparison(left, right, output)
    pair_text = (output / "public" / "blind_pairs.json").read_text(encoding="utf-8")
    pairs = json.loads(pair_text)
    assert pairs[0]["pair_id"] == "case_one__run-01"
    assert set(pairs[0]) == {"pair_id", "case_id", "run_index", "candidate_a", "candidate_b"}
    assert "0.1" not in pair_text and "0.2" not in pair_text
    mapping_text = (output / "private" / "candidate_mapping.json").read_text(encoding="utf-8")
    assert "0.1" in mapping_text and "0.2" in mapping_text
    assert "prompt-0.1" in mapping_text and "prompt-0.2" in mapping_text
    assert "teacher_preference" not in pair_text
    assert "learning_effect" not in pair_text


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
