import asyncio
import json
from pathlib import Path
import sys
from typing import List


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.audio_service import LessonAudioService
from app.config import Settings
from app.generation import LessonGenerationService
from app.llm_client import OpenAICompatibleClient
from app.math_engine import MathEngine
from app.schemas import ProblemInput
from app.tts_client import OpenAISpeechClient
from app.volcengine_tts_client import VolcengineSpeechClient


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


async def main() -> None:
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

    model_client = OpenAICompatibleClient(settings)
    speech_client = create_speech_client(settings)
    try:
        lesson = await LessonGenerationService(
            model_client,
            MathEngine(),
        ).generate(
            ProblemInput(
                problem_text="用配方法解方程：x^2-6x+5=0",
                reference_answer="x=1 或 x=5",
                reference_solution_text=(
                    "解：移项，得 x^2-6x=-5。\n\n"
                    "两边同时加9，得 (x-3)^2=4。\n"
                    "所以 x-3=2 或 x-3=-2，\n"
                    "即 x=5 或 x=1。"
                ),
                required_method="complete_the_square",
            )
        )
        lesson = await LessonAudioService(
            speech_client,
            REPOSITORY_ROOT / "var" / "audio",
        ).attach_audio(lesson)
        report = lesson.validation_report
        print(
            json.dumps(
                {
                    "lesson_id": lesson.lesson_id,
                    "beat_count": len(lesson.beats),
                    "audio_ready": all(
                        bool(beat.audio_url) for beat in lesson.beats
                    ),
                    "math_status": report.get("math_status"),
                    "review_status": report.get("review_status"),
                    "reference_material_status": report.get(
                        "reference_material_status"
                    ),
                    "revision_count": report.get("revision_count"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await model_client.close()
        await speech_client.close()


if __name__ == "__main__":
    asyncio.run(main())
