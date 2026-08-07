from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import ApiServices, create_api_router
from app.audio_service import LessonAudioService
from app.config import Settings
from app.llm_client import OpenAICompatibleClient
from app.math_engine import MathEngine
from app.store import MemoryStore
from app.tts_client import OpenAISpeechClient
from app.volcengine_tts_client import VolcengineSpeechClient


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
STATIC_ROOT = APP_ROOT / "static"
AUDIO_ROOT = PROJECT_ROOT / "var" / "audio"


def create_app(
    settings: Optional[Settings] = None,
    generator: Any = None,
    audio_service: Any = None,
    store: Optional[MemoryStore] = None,
    math_engine: Optional[MathEngine] = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    services = ApiServices(
        settings=resolved_settings,
        store=store or MemoryStore(),
        math_engine=math_engine or MathEngine(),
        generator=generator,
        audio_service=audio_service,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        owned_clients: List[Any] = []
        owns_generator = False
        owns_audio_service = False

        if services.generator is None:
            from app.generation import LessonGenerationService

            model_client = OpenAICompatibleClient(resolved_settings)
            owned_clients.append(model_client)
            services.generator = LessonGenerationService(
                model_client,
                services.math_engine,
            )
            owns_generator = True

        if services.audio_service is None:
            if resolved_settings.tts_provider == "volcengine":
                speech_client = VolcengineSpeechClient(resolved_settings)
            else:
                speech_client = OpenAISpeechClient(resolved_settings)
            owned_clients.append(speech_client)
            services.audio_service = LessonAudioService(
                speech_client,
                AUDIO_ROOT,
            )
            owns_audio_service = True

        try:
            yield
        finally:
            closed = set()
            for client in reversed(owned_clients):
                identity = id(client)
                if identity in closed:
                    continue
                closed.add(identity)
                await client.close()
            if owns_generator:
                services.generator = None
            if owns_audio_service:
                services.audio_service = None

    application = FastAPI(lifespan=lifespan)
    application.state.services = services
    application.state.store = services.store
    application.include_router(create_api_router(services))
    application.mount(
        "/static",
        StaticFiles(directory=STATIC_ROOT, check_dir=False),
        name="static",
    )
    application.mount(
        "/audio",
        StaticFiles(directory=AUDIO_ROOT, check_dir=False),
        name="audio",
    )

    @application.get("/", include_in_schema=False)
    async def generation_page():
        page = STATIC_ROOT / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(page)

    @application.get("/lesson/{lesson_id}", include_in_schema=False)
    async def lesson_page(lesson_id: str):
        del lesson_id
        page = STATIC_ROOT / "lesson.html"
        if not page.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(
            page,
            headers={"Cache-Control": "no-cache"},
        )

    return application


app = create_app()
