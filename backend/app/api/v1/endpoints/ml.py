"""ML model prediction endpoints via Hugging Face Space."""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.hf_client import call_hf_predict, HFClientError

_LOGGER = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    """Request schema for ML prediction."""
    input: str = Field(..., min_length=1, max_length=5000, description="User symptom input text")


class PredictResponse(BaseModel):
    """Response schema for ML prediction."""
    status: str
    data: dict | list | str | None = None
    message: str | None = None
    details: str | None = None


@router.post("/predict", response_model=PredictResponse, summary="Get ML model prediction from HF Space")
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Get model prediction from Hugging Face Space.
    
    Sends user input to the HF Space ML model and returns the prediction result.
    
    Args:
        payload: Request containing symptom input text
        
    Returns:
        dict: Contains status, prediction data, and any error message
        
    Raises:
        HTTPException: If HF Space API fails
    """
    try:
        _LOGGER.info(f"[ML Endpoint] Prediction request received. Input length: {len(payload.input)}")
        _LOGGER.info(f"[ML Endpoint] Calling HF Space...")
        
        hf_response = await call_hf_predict(payload.input)
        
        _LOGGER.info(f"[ML Endpoint] HF Space returned successfully. Response type: {type(hf_response)}")
        
        # Return successful response with HF data
        return PredictResponse(
            status="success",
            data=hf_response,
            message=None,
            details=None
        )
        
    except HFClientError as exc:
        _LOGGER.error("[ML Endpoint] HF Client error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "ML service unavailable"},
        )
        
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        _LOGGER.error(f"[ML Endpoint] Unexpected error: {error_msg}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Failed to process prediction request",
                "details": error_msg
            }
        ) from exc

