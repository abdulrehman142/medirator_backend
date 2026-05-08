from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import os
from app.services.medgemma_service import MedGemmaMultimodalService

router = APIRouter()

# Lazy singleton for the AI engine. Model and heavy deps are loaded on demand.
_ai_engine: MedGemmaMultimodalService | None = None


def get_ai_engine() -> MedGemmaMultimodalService:
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = MedGemmaMultimodalService()
    return _ai_engine


@router.post("/process")
async def handle_query(
    prompt: str = Form(...),
    image: UploadFile = File(None),
):
    temp_path = None
    if image:
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    try:
        ai_engine = get_ai_engine()
        # load_model will be called automatically if needed inside generate()
        response = ai_engine.generate(text=prompt, image_path=temp_path)

        if temp_path:
            os.remove(temp_path)
        return {"result": response}

    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))