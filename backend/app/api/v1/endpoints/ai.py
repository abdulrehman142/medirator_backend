"""AI model inference endpoints."""

import logging
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, status

from app.services.hf_client import call_hf_predict, HFClientError

_LOGGER = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process")
async def handle_query(prompt: str = Form(...), image: UploadFile = File(None)):
    """
    Process AI query with optional image input via Hugging Face Space model.
    
    Args:
        prompt: Text prompt/query
        image: Optional uploaded image file
        
    Returns:
        dict: HF Space model response
    """
    try:
        # Combine prompt and image filename if image is provided
        input_text = prompt
        if image:
            input_text = f"{prompt} [Image: {image.filename}]"
        
        _LOGGER.info(f"[AI Endpoint] Processing query. Length: {len(input_text)}")
        
        # Call HF Space API for inference
        result = call_hf_predict(input_text)
        
        _LOGGER.info(f"[AI Endpoint] HF Space inference successful")
        
        return {
            "status": "success",
            "response": result,
            "model": "hf_space"
        }
        
    except HFClientError as exc:
        _LOGGER.error(f"[AI Endpoint] HF Client error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "message": "Failed to connect to AI model",
                "details": str(exc)
            }
        ) from exc
        
    except Exception as exc:
        _LOGGER.error(f"[AI Endpoint] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "AI inference failed",
                "details": str(exc)
            }
        ) from exc
