import asyncio
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import httpx


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.audio_service import LessonAudioService
from app.config import Settings
from app.generation import LessonGenerationService, LessonQualityError
from app.llm_client import ModelResponseError, OpenAICompatibleClient
from app.math_engine import MathEngine
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
from app.preparation_pipeline import (
    LessonPreparationPipeline,
    PreparationFailure,
)
from app.schemas import ProblemInput
from app.tts_client import (
    OpenAISpeechClient,
    SpeechGenerationError,
)
from app.volcengine_tts_client import VolcengineSpeechClient


REFERENCE_SOLUTION_TEXT = (
    "解：移项，得 x^2-6x=-5。\n\n"
    "两边同时加9，得 (x-3)^2=4。\n"
    "所以 x-3=2 或 x-3=-2，\n"
    "即 x=5 或 x=1。"
)
GROUNDED_PARAMETER_ROOT_PROBLEM = (
    "若$2n$ ($n\\ne 0$)是关于 x的方程 "
    "$x^2-2mx+2n=0$的根，则m-n的值为"
)
GROUNDED_PARAMETER_ROOT_ANSWER = r"$\frac{1}{2}$"
GROUNDED_PARAMETER_ROOT_SOLUTION = (
    "因为 $2n(n\\ne 0)$ 是关于x的方程"
    "$x^2-2mx+2n=0$的解\n"
    "所以 $4n^2-4mn+2n=0$\n"
    "所以$4n-4m+2=0$\n"
    "所以$m-n=\\frac{1}{2}$"
)


class SmokeContractError(RuntimeError):
    """Raised when a live lesson misses a safe, structural smoke contract."""


class RecordingModelClient:
    """Record prompt identities for smoke assertions without logging content."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.system_prompts = []

    async def complete_json(self, system_prompt, user_prompt):
        self.system_prompts.append(system_prompt)
        return await self.delegate.complete_json(system_prompt, user_prompt)

    async def complete_model_with_metadata(
        self,
        system_prompt,
        user_prompt,
        model_type,
    ):
        self.system_prompts.append(system_prompt)
        structured_method = getattr(
            self.delegate,
            "complete_model_with_metadata",
            None,
        )
        if callable(structured_method):
            return await structured_method(
                system_prompt,
                user_prompt,
                model_type,
            )
        return await self.delegate.complete_json(system_prompt, user_prompt)

    async def close(self):
        return await self.delegate.close()


def _require_contract(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeContractError(message)


PromptRun = Tuple[str, int]
_PREPARATION_PROMPT_STAGES = {
    SOLUTION_TRACE_SYSTEM: "TRACE",
    TEACHING_DESIGNER_SYSTEM: "TRAJECTORY",
    TEACHING_PROGRESSION_SYSTEM: "PROGRESSION",
    INTERACTION_DESIGNER_SYSTEM: "INTERACTION",
    SCRIPT_TEACHER_SYSTEM: "SCRIPT",
    CLASSROOM_DIRECTOR_SYSTEM: "PERFORMANCE",
    STUDENT_SIMULATOR_SYSTEM: "SIMULATION",
    LESSON_REVIEWER_SYSTEM: "REVIEW",
}
_CORE_PROMPT_STAGES = {
    REFERENCE_AUDITOR_SYSTEM: "A",
    **_PREPARATION_PROMPT_STAGES,
}
_GROUNDED_PROMPT_STAGES = {
    REFERENCE_GROUNDING_SYSTEM: "G",
    **_PREPARATION_PROMPT_STAGES,
}


def _prompt_runs(
    system_prompts: Sequence[str],
    *,
    prompt_stages: Dict[str, str],
    unknown_message: str,
) -> List[PromptRun]:
    runs: List[PromptRun] = []
    for prompt in system_prompts:
        stage = prompt_stages.get(prompt)
        _require_contract(stage is not None, unknown_message)
        assert stage is not None
        if runs and runs[-1][0] == stage:
            runs[-1] = (stage, runs[-1][1] + 1)
        else:
            runs.append((stage, 1))
    return runs


def assert_model_call_contract(
    system_prompts: Sequence[str],
    grounded_parameter_root: bool = False,
    with_reference_audit: bool = False,
) -> None:
    _require_contract(
        MATH_ROUTE_SYSTEM not in system_prompts,
        "配方法 smoke 未使用确定性数学路线。",
    )
    runs = _prompt_runs(
        system_prompts,
        prompt_stages=(
            _GROUNDED_PROMPT_STAGES
            if grounded_parameter_root
            else _CORE_PROMPT_STAGES
        ),
        unknown_message="smoke 出现未知 Agent 调用。",
    )
    prefix = []
    if grounded_parameter_root:
        prefix = ["G"]
    elif with_reference_audit:
        prefix = ["A"]
    stages = [stage for stage, _count in runs]
    _require_contract(
        stages[: len(prefix)] == prefix,
        "smoke 的路线审计阶段顺序不正确。",
    )
    _require_contract(
        all(
            count <= LessonGenerationService.MAX_GROUNDING_ATTEMPTS
            for _stage, count in runs[: len(prefix)]
        ),
        "smoke 的路线审计阶段重试超限。",
    )
    preparation_order = [
        "TRACE",
        "TRAJECTORY",
        "PROGRESSION",
        "INTERACTION",
        "SCRIPT",
        "PERFORMANCE",
        "SIMULATION",
        "REVIEW",
    ]
    preparation_runs = runs[len(prefix) :]
    initial_runs = preparation_runs[: len(preparation_order)]
    _require_contract(
        [stage for stage, _count in initial_runs]
        == preparation_order,
        "smoke 未按依赖顺序完成首轮备课、模拟与审核。",
    )
    _require_contract(
        all(
            count <= LessonPreparationPipeline.MAX_STRUCTURE_ATTEMPTS
            for _stage, count in initial_runs
        ),
        "smoke 首轮备课或审核的结构重试超限。",
    )
    remaining = preparation_runs[len(preparation_order) :]
    repair_starts = preparation_order[:6]
    repair_cycles = 0
    while remaining:
        repair_cycles += 1
        _require_contract(
            repair_cycles <= LessonPreparationPipeline.MAX_REPAIR_CYCLES,
            "smoke 的定向修复轮数超限。",
        )
        repair_start = remaining[0][0]
        _require_contract(
            repair_start in repair_starts,
            "smoke 出现非法的定向修复起点。",
        )
        start_index = preparation_order.index(repair_start)
        expected_suffix = preparation_order[start_index:]
        actual_suffix = remaining[: len(expected_suffix)]
        _require_contract(
            [stage for stage, _count in actual_suffix]
            == expected_suffix,
            "smoke 的定向修复未完整重建下游并审核。",
        )
        _require_contract(
            all(
                count <= LessonPreparationPipeline.MAX_STRUCTURE_ATTEMPTS
                for _stage, count in actual_suffix[:-1]
            ),
            "smoke 的定向修复角色结构重试超限。",
        )
        _require_contract(
            actual_suffix[-1][1]
            <= 2 * LessonPreparationPipeline.MAX_STRUCTURE_ATTEMPTS,
            "smoke 定向修复后的审核重试超限。",
        )
        remaining = remaining[len(expected_suffix) :]


def assert_common_lesson_contract(
    lesson,
    *,
    require_audio: bool = True,
) -> dict:
    """Assert shared lesson structure without using symbolic math tools."""
    beats = lesson.beats
    _require_contract(
        bool(beats),
        "生成课程没有可播放的节拍。",
    )

    interactions = [
        beat.interaction for beat in beats if beat.interaction is not None
    ]
    interaction_kinds = [interaction.kind for interaction in interactions]
    _require_contract(
        bool(interactions),
        "生成课程没有互动。",
    )
    _require_contract(
        all(kind == "choice" for kind in interaction_kinds),
        "生成课程未保持选择式互动。",
    )

    transfer_interaction = next(
        (
            item
            for item in interactions
            if getattr(item, "interaction_id", None) == "near-transfer"
        ),
        None,
    )
    choices = [
        item
        for item in interactions
        if getattr(item, "interaction_id", None) != "near-transfer"
    ]
    _require_contract(bool(choices), "生成课程没有讲解中的诊断互动。")
    for choice in choices:
        _require_contract(
            len(choice.options) in {3, 4},
            "生成选择互动的选项数量不符合 smoke 合同。",
        )
        for option in choice.options:
            if option.option_id == choice.expected_answer:
                _require_contract(
                    not getattr(option, "support_cues", []),
                    "正确选项不应重复播放主线讲解。",
                )
                continue
            support_cues = getattr(option, "support_cues", [])
            _require_contract(
                bool(getattr(option, "feedback", None)) or bool(support_cues),
                "生成选择互动缺少错误诊断反馈。",
            )
            _require_contract(
                bool(getattr(option, "feedback_audio_url", None))
                or all(bool(cue.audio_url) for cue in support_cues),
                "生成选择互动缺少错误诊断反馈语音。",
            )

    if transfer_interaction is not None:
        transfer_options = lesson.transfer_item.options
        _require_contract(
            len(transfer_interaction.options) == len(transfer_options),
            "近迁移选项数量与内部课程记录不一致。",
        )
        for transfer_option, runtime_option in zip(
            transfer_options,
            transfer_interaction.options,
        ):
            _require_contract(
                transfer_option.option_id == runtime_option.option_id,
                "近迁移选项顺序或标识与内部课程记录不一致。",
            )
            _require_contract(
                runtime_option.label == transfer_option.label,
                "近迁移选项显示标签与内部课程记录不一致。",
            )

    audio_ready = all(
        all(bool(cue.audio_url) for cue in beat.sync_cues)
        if getattr(beat, "sync_cues", None)
        else bool(beat.audio_url)
        for beat in beats
    )
    if require_audio:
        _require_contract(audio_ready, "生成课程缺少讲解语音。")
    return {
        "interaction_kinds": interaction_kinds,
        "diagnostic_choice_count": len(choices),
        "option_feedback_audio_ready": True,
        "audio_ready": audio_ready,
    }


def assert_generated_lesson_contract(
    lesson,
    math_engine: MathEngine = None,
) -> dict:
    """Assert the core complete-the-square smoke contract."""
    common = assert_common_lesson_contract(lesson)
    beats = lesson.beats
    _require_contract(
        len(beats) >= 2,
        "生成课程缺少方法介绍节拍。",
    )
    method_beat = beats[1]
    _require_contract(
        method_beat.purpose == "先认识方法"
        and method_beat.layer == "micro_explanation",
        "生成课程未以方法介绍作为第二个节拍。",
    )
    _require_contract(
        bool(method_beat.board_actions)
        and method_beat.board_actions[0].type == "write"
        and method_beat.board_actions[0].target == "method_name"
        and method_beat.board_actions[0].content == "配方法",
        "方法介绍的首个板书动作不符合配方法 smoke 合同。",
    )
    _require_contract(
        method_beat.narration.startswith("今天用配方法"),
        "方法介绍口语讲稿未以配方法开头。",
    )
    _require_contract(
        "\\" not in method_beat.narration,
        "方法介绍口语讲稿含有反斜杠。",
    )

    engine = math_engine or MathEngine()
    transfer_interaction = next(
        (
            beat.interaction
            for beat in beats
            if beat.interaction is not None
            and getattr(beat.interaction, "interaction_id", None)
            == "near-transfer"
        ),
        None,
    )
    runtime_transfer_options = (
        transfer_interaction.options if transfer_interaction is not None else []
    )
    for index, transfer_option in enumerate(lesson.transfer_item.options):
        try:
            expected_label = engine.format_answer_label(
                transfer_option.canonical_answer
            )
        except Exception:
            raise SmokeContractError(
                "近迁移选项的内部答案无法生成显示标签。"
            ) from None
        _require_contract(
            transfer_option.label == expected_label
            and (
                not runtime_transfer_options
                or runtime_transfer_options[index].label == expected_label
            ),
            "近迁移选项标签与内部答案不一致。",
        )

    return {
        "method_first": True,
        **common,
        "formula_labels_ready": True,
    }


def _strip_outer_math_delimiters(value: str) -> str:
    text = value.strip()
    delimiters = (
        ("$$", "$$"),
        ("$", "$"),
        (r"\(", r"\)"),
        (r"\[", r"\]"),
    )
    changed = True
    while changed:
        changed = False
        for opening, closing in delimiters:
            if (
                text.startswith(opening)
                and text.endswith(closing)
                and len(text) > len(opening) + len(closing)
            ):
                text = text[len(opening) : -len(closing)].strip()
                changed = True
                break
    return text


def _parse_half_expression(value: str) -> Optional[Fraction]:
    latex_fraction = re.fullmatch(
        r"\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}",
        value,
    )
    short_latex_fraction = re.fullmatch(
        r"\\frac([+-]?\d)([+-]?\d)",
        value,
    )
    plain_fraction = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", value)
    match = latex_fraction or short_latex_fraction or plain_fraction
    try:
        if match:
            return Fraction(int(match.group(1)), int(match.group(2)))
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", value):
            return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _is_parameter_root_conclusion(value: str) -> bool:
    text = _strip_outer_math_delimiters(value)
    rejected_markers = (
        "错误",
        "错解",
        "不等于",
        "不是",
        "并非",
        "!=",
        "≠",
        "≤",
        "≥",
        "<",
        ">",
        r"\ne",
        r"\neq",
        r"\le",
        r"\ge",
    )
    if any(marker in text for marker in rejected_markers):
        return False
    compact = re.sub(r"\s+", "", text).replace("−", "-")
    if compact.count("=") != 1:
        return False
    left, right = compact.split("=", 1)
    if left != "m-n":
        return False
    return _parse_half_expression(right) == Fraction(1, 2)


def _is_parameter_root_substitution(value: str) -> bool:
    if not isinstance(value, str):
        return False
    compact = re.sub(
        r"\s+",
        "",
        _strip_outer_math_delimiters(value),
    ).replace("−", "-")
    for spacing_command in (r"\,", r"\;", r"\!"):
        compact = compact.replace(spacing_command, "")
    compact = compact.replace("²", "^2").replace("^{2}", "^2")
    return compact == "4n^2-4mn+2n=0"


def assert_grounded_parameter_root_contract(lesson) -> dict:
    """Check a grounded live lesson without returning private lesson content."""
    common = assert_common_lesson_contract(lesson, require_audio=False)
    _require_contract(
        all(beat.sync_cues for beat in lesson.beats),
        "参数根课程并非每个 Beat 都有同步 Cue。",
    )
    cues = [
        cue
        for beat in lesson.beats
        for cue in beat.sync_cues
    ]
    cue_audio_ready = all(bool(cue.audio_url) for cue in cues)
    _require_contract(
        cue_audio_ready,
        "参数根课程缺少 Cue 语音。",
    )
    _require_contract(
        any(
            action.surface == "problem"
            and action.type == "emphasize"
            and action.target == "problem-math-001"
            for cue in cues
            for action in cue.lead_actions
        ),
        "参数根课程缺少第一个题目公式的 Cue lead 强调。",
    )
    _require_contract(
        any(
            action.surface == "board"
            and action.type in {"write", "transform"}
            and _is_parameter_root_substitution(action.content)
            for cue in cues
            for action in cue.start_actions
        ),
        "参数根课程缺少包含 4n 的 Cue start 板书。",
    )
    report = lesson.validation_report
    mode = report.get("verification_mode")
    _require_contract(
        mode in {"model_cross_checked", "reference_grounded"},
        "参数根课程验证模式不符合 grounded smoke 合同。",
    )
    _require_contract(
        report.get("consistency_status") in {"consistent", "warning"},
        "参数根课程缺少可接受的一致性状态。",
    )
    _require_contract(
        bool(report.get("teaching_route_fingerprint")),
        "参数根课程缺少冻结教学路线指纹。",
    )
    review_status = report.get("review_status")
    _require_contract(
        review_status == "approved",
        "参数根课程未通过整篇审稿。",
    )

    board_contents = (
        action.content
        for beat in lesson.beats
        for action in beat.board_actions
        if action.type in {"write", "transform"}
        and isinstance(action.content, str)
    )
    conclusion_present = any(
        _is_parameter_root_conclusion(content) for content in board_contents
    )
    _require_contract(
        conclusion_present,
        "参数根课程板书缺少参考结论。",
    )

    return {
        "lesson_id": lesson.lesson_id,
        "beat_count": len(lesson.beats),
        "cue_count": len(cues),
        "interaction_kinds": common["interaction_kinds"],
        "review_status": review_status,
        "audio_ready": cue_audio_ready,
        "conclusion_present": conclusion_present,
    }


def missing_environment(settings: Settings) -> List[str]:
    missing = list(settings.missing_model_settings)
    if settings.tts_provider == "volcengine":
        voice_settings = (
            ("VOLCENGINE_TTS_ENDPOINT", settings.volcengine_tts_endpoint),
            (
                "VOLCENGINE_TTS_RESOURCE_ID",
                settings.volcengine_tts_resource_id,
            ),
            ("VOLCENGINE_TTS_VOICE", settings.volcengine_tts_voice),
            ("VOLCENGINE_TTS_UID", settings.volcengine_tts_uid),
        )
        missing.extend(
            name
            for name, value in voice_settings
            if not isinstance(value, str) or not value.strip()
        )
        if not settings.volcengine_tts_api_key:
            missing.append("VOLCENGINE_TTS_API_KEY")
        return missing

    voice_settings = (
        ("TTS_BASE_URL 或 OPENAI_BASE_URL", settings.tts_base_url),
        ("TTS_API_KEY 或 OPENAI_API_KEY", settings.tts_api_key),
        ("TTS_MODEL", settings.tts_model),
        ("TTS_VOICE", settings.tts_voice),
    )
    missing.extend(
        name
        for name, value in voice_settings
        if not isinstance(value, str) or not value.strip()
    )
    return missing


def create_speech_client(settings: Settings):
    if settings.tts_provider == "volcengine":
        return VolcengineSpeechClient(settings)
    return OpenAISpeechClient(settings)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the safe live smoke for the AI math lesson runtime."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--with-reference-audit",
        action="store_true",
        help="Also exercise the optional reference-material audit stage.",
    )
    mode.add_argument(
        "--grounded-parameter-root",
        action="store_true",
        help="Exercise the reference-grounded parameter-root lesson.",
    )
    return parser.parse_args(argv)


def smoke_problem(
    with_reference_audit: bool,
    grounded_parameter_root: bool = False,
) -> ProblemInput:
    if grounded_parameter_root:
        return ProblemInput(
            problem_text=GROUNDED_PARAMETER_ROOT_PROBLEM,
            reference_answer=GROUNDED_PARAMETER_ROOT_ANSWER,
            reference_solution_text=GROUNDED_PARAMETER_ROOT_SOLUTION,
        )
    return ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        reference_solution_text=(
            REFERENCE_SOLUTION_TEXT if with_reference_audit else None
        ),
        required_method="complete_the_square",
    )


async def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        settings = Settings.from_env()
    except ValueError as error:
        if str(error).startswith("OPENAI_TIMEOUT_SECONDS"):
            message = "OPENAI_TIMEOUT_SECONDS 配置无效。"
        else:
            message = "TTS 环境变量配置无效。"
        raise SystemExit(
            message + "未发起网络请求。"
        ) from None
    missing = missing_environment(settings)
    if missing:
        raise SystemExit(
            "缺少环境变量：" + "、".join(missing) + "。未发起网络请求。"
        )

    model_client = None
    speech_client = None
    primary_error = None
    try:
        model_client = RecordingModelClient(
            OpenAICompatibleClient(settings)
        )
        speech_client = create_speech_client(settings)
        math_engine = MathEngine()
        with tempfile.TemporaryDirectory(
            prefix="ai-math-smoke-audio-"
        ) as temporary_audio_root:
            lesson = await LessonGenerationService(
                model_client,
                math_engine,
            ).generate(
                smoke_problem(
                    args.with_reference_audit,
                    args.grounded_parameter_root,
                )
            )
            assert_model_call_contract(
                model_client.system_prompts,
                grounded_parameter_root=args.grounded_parameter_root,
                with_reference_audit=args.with_reference_audit,
            )
            lesson = await LessonAudioService(
                speech_client,
                Path(temporary_audio_root),
            ).attach_audio(lesson)
            if args.grounded_parameter_root:
                summary = assert_grounded_parameter_root_contract(
                    lesson
                )
            else:
                smoke_contract = assert_generated_lesson_contract(
                    lesson,
                    math_engine,
                )
                report = lesson.validation_report
                if args.with_reference_audit:
                    _require_contract(
                        report.get("reference_material_status")
                        == "approved",
                        "可选参考解析审阅未通过 smoke 合同。",
                    )
                summary = {
                    "mode": (
                        "reference_audit"
                        if args.with_reference_audit
                        else "core"
                    ),
                    "lesson_id": lesson.lesson_id,
                    "beat_count": len(lesson.beats),
                    **smoke_contract,
                    "math_status": report.get("math_status"),
                    "review_status": report.get("review_status"),
                    "repair_count": report.get("repair_count"),
                    "math_route_source": report.get("math_route_source"),
                }
                if args.with_reference_audit:
                    summary["reference_material_status"] = report.get(
                        "reference_material_status"
                    )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = None
        for client in (speech_client, model_client):
            if client is None:
                continue
            try:
                await client.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if primary_error is None and close_error is not None:
            raise close_error

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


def run_cli(argv=None) -> None:
    try:
        asyncio.run(main(argv))
    except ModelResponseError:
        raise SystemExit(
            "模型服务调用失败，请检查配置或稍后重试。"
        ) from None
    except PreparationFailure as error:
        if error.category == "provider_error":
            raise SystemExit(
                "模型服务调用失败，请检查配置或稍后重试。"
            ) from None
        raise SystemExit(
            "讲解生成未通过质量门，请检查模型输出后重试。"
        ) from None
    except LessonQualityError:
        raise SystemExit(
            "讲解生成未通过质量门，请检查模型输出后重试。"
        ) from None
    except SpeechGenerationError:
        raise SystemExit(
            "语音生成失败，请检查 TTS 配置或稍后重试。"
        ) from None
    except httpx.HTTPError:
        raise SystemExit(
            "现场服务网络请求失败，请检查网络连接或稍后重试。"
        ) from None
    except SmokeContractError:
        raise SystemExit("现场课程未通过 smoke 合同检查。") from None


if __name__ == "__main__":
    run_cli()
