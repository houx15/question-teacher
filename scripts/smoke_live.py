import asyncio
import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import List

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
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REVIEWER_SYSTEM,
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

    async def close(self):
        return await self.delegate.close()


def _require_contract(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeContractError(message)


def assert_model_call_contract(system_prompts) -> None:
    _require_contract(
        MATH_ROUTE_SYSTEM not in system_prompts,
        "配方法 smoke 未使用确定性数学路线。",
    )
    core_calls = [
        prompt
        for prompt in system_prompts
        if prompt in {
            DIRECTOR_SYSTEM,
            MATERIALS_SYSTEM,
            REVIEWER_SYSTEM,
        }
    ]
    _require_contract(
        core_calls[:3]
        == [
            DIRECTOR_SYSTEM,
            MATERIALS_SYSTEM,
            REVIEWER_SYSTEM,
        ],
        "核心教学 Agent 调用顺序不符合 smoke 合同。",
    )


def assert_generated_lesson_contract(
    lesson,
    math_engine: MathEngine = None,
) -> dict:
    """Assert structural teaching and audio guarantees without exposing content."""
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

    interactions = [
        beat.interaction for beat in beats if beat.interaction is not None
    ]
    interaction_kinds = [interaction.kind for interaction in interactions]
    _require_contract(
        not {"expression", "transfer"}.intersection(interaction_kinds),
        "新生成课程包含已弃用的数学输入互动。",
    )

    choices = [
        interaction
        for interaction in interactions
        if interaction.kind == "choice"
    ]
    for choice in choices:
        _require_contract(
            len(choice.options) in {3, 4},
            "生成选择互动的选项数量不符合 smoke 合同。",
        )
        for option in choice.options:
            _require_contract(
                bool(option.feedback),
                "生成选择互动缺少诊断反馈。",
            )
            _require_contract(
                bool(option.feedback_audio_url),
                "生成选择互动缺少诊断反馈语音。",
            )

    final_interaction = beats[-1].interaction
    _require_contract(
        final_interaction is not None
        and final_interaction.kind == "choice",
        "生成课程未以选择式近迁移互动结束。",
    )
    transfer_options = lesson.transfer_item.options
    _require_contract(
        len(final_interaction.options) == len(transfer_options),
        "近迁移选项数量与内部课程记录不一致。",
    )
    engine = math_engine or MathEngine()
    for transfer_option, runtime_option in zip(
        transfer_options,
        final_interaction.options,
    ):
        _require_contract(
            transfer_option.option_id == runtime_option.option_id,
            "近迁移选项顺序或标识与内部课程记录不一致。",
        )
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
            and runtime_option.label == expected_label,
            "近迁移选项标签与内部答案不一致。",
        )

    audio_ready = all(bool(beat.audio_url) for beat in beats)
    _require_contract(audio_ready, "生成课程缺少讲解语音。")
    return {
        "method_first": True,
        "interaction_kinds": interaction_kinds,
        "diagnostic_choice_count": len(choices),
        "option_feedback_audio_ready": True,
        "formula_labels_ready": True,
        "audio_ready": audio_ready,
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
    parser.add_argument(
        "--with-reference-audit",
        action="store_true",
        help="Also exercise the optional reference-material audit stage.",
    )
    return parser.parse_args(argv)


def smoke_problem(with_reference_audit: bool) -> ProblemInput:
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
            ).generate(smoke_problem(args.with_reference_audit))
            assert_model_call_contract(model_client.system_prompts)
            lesson = await LessonAudioService(
                speech_client,
                Path(temporary_audio_root),
            ).attach_audio(lesson)
            smoke_contract = assert_generated_lesson_contract(
                lesson,
                math_engine,
            )
            report = lesson.validation_report
            if args.with_reference_audit:
                _require_contract(
                    report.get("reference_material_status") == "approved",
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
                "revision_count": report.get("revision_count"),
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
