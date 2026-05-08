from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.api.v1.endpoints import chat as chat_endpoints
from app.schemas.ai import ChatResponse
from app.vision import maybe_image_path
from app.voice import transcribe_audio_file
from app.core.config import get_settings
from app.db.indexes import ensure_indexes
from app.db.mongo import close_mongo, init_mongo
from app.db.redis import close_redis
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_log import RequestLogMiddleware
from app.utils.logger import configure_logging

# Ensure .env is loaded even when uvicorn runs
load_dotenv()

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    db = init_mongo()
    await ensure_indexes(db)
    
    yield
    await close_mongo()
    await close_redis()


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestLogMiddleware)


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_UPLOAD_TEMP = _BACKEND_ROOT / "temp"


def _ensure_temp_dir() -> None:
    _UPLOAD_TEMP.mkdir(parents=True, exist_ok=True)


chat_root_router = APIRouter(tags=["chat"])


@chat_root_router.post("/chat", response_model=ChatResponse)
async def multimodal_root_chat(
    request: Request,
    text: Optional[str] = Form(None),
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    voice: Optional[UploadFile] = File(None),
) -> ChatResponse:
    """
    Top-level multimodal chat:
    optional image upload, optional voice (converted to text), and text/message body.
    """
    user = await chat_endpoints.get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to use MediBot chatbot.")

    _ensure_temp_dir()
    content_parts = [(text or "").strip(), (message or "").strip()]
    content = "\n".join(part for part in content_parts if part).strip()

    persisted: list[Path] = []
    image_model_path: str | None = None

    try:
        async def persist_upload(upload: UploadFile, stem: str) -> Path:
            suffix = Path(upload.filename or "").suffix or ""
            dest = _UPLOAD_TEMP / f"{stem}_{uuid.uuid4().hex}{suffix}"
            data = await upload.read()
            dest.write_bytes(data)
            persisted.append(dest)
            return dest

        if voice is not None:
            vp = await persist_upload(voice, "voice")
            try:
                spoken = transcribe_audio_file(vp)
            except RuntimeError:
                spoken = ""
            if spoken:
                content = f"{content}\n{spoken}".strip() if content else spoken.strip()

        if image is not None:
            ipath = await persist_upload(image, "image")
            path_str, ok = maybe_image_path(image.filename, str(ipath))
            if ok and path_str:
                image_model_path = path_str

        if not content.strip():
            raise HTTPException(
                status_code=400,
                detail="Provide `text` or `message`, or upload `voice` that can be transcribed.",
            )

        result = await chat_endpoints.ai_orchestrator.process_interaction(
            text=content.strip(),
            image_path=image_model_path,
            conversation_history=None,
            uploaded_file_contexts=None,
        )
        return ChatResponse(
            reply=result.get("answer", ""),
            model_used=result.get("model_used"),
            confidence=result.get("confidence"),
            disclaimer=result.get("disclaimer"),
        )
    finally:
        for path in persisted:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_root_router)
app.include_router(api_router, prefix=settings.api_v1_prefix)
