from fastapi import APIRouter, Form, UploadFile, File

router = APIRouter()


@router.post("/process")
async def handle_query(prompt: str = Form(...), image: UploadFile = File(None)):
    """ML execution disabled in backend for deployment.

    Returns a standardized placeholder message so the API remains
    functional without heavy ML dependencies.
    """
    return {"message": "model_service_not_available"}