"""X-ray analysis endpoint backed by Hugging Face Space."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.services.hf_client import HFClientError
from app.services.xrayas_service import xrayas_service_instance

router = APIRouter()


@router.post("/xray-analysis")
async def xray_analysis(
    image: UploadFile = File(...),
    notes: str | None = Form(default=None),
) -> dict:
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        image_bytes = await image.read()
        result = await xrayas_service_instance.analyze(image.filename, notes, image_bytes)
        return {
            "status": "success",
            "prediction": result.get("raw", result),
            "confidence": result.get("confidence", 0.0),
            "answer": result.get("answer", ""),
        }
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
