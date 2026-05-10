"""X-ray analysis endpoint backed by Hugging Face Space."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.services.hf_client import HFClientError
from app.services.xrayas_service import xrayas_service_instance

router = APIRouter()


@router.post("/xray-analysis")
async def xray_analysis(
    image: UploadFile = File(...),
    notes: str | None = Form(default=None),
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        image_bytes = await image.read()
        result = await xrayas_service_instance.analyze(image.filename, notes, image_bytes)
        
        # Extract findings and confidence
        answer = result.get("answer", "Unable to analyze image")
        confidence = result.get("confidence", 0.0)
        confidence_pct = round(confidence * 100, 1)
        
        # Format text response
        formatted_response = f"X-Ray Analysis:\n{answer}\nConfidence: {confidence_pct}%"
        return PlainTextResponse(content=formatted_response)
    except HFClientError as exc:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "ML service unavailable"},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": str(exc)},
        ) from exc
