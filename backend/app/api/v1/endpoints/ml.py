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
    prediction: str | None = None
    confidence: float | None = None
    details: str | None = None


def _parse_hf_response(hf_data: list | dict) -> tuple[str | None, dict | None]:
    """
    Parse HF Space response into prediction and xray results.
    
    Args:
        hf_data: Raw response from HF Space (list or dict)
        
    Returns:
        tuple: (prediction_string, xray_results_dict)
    """
    if isinstance(hf_data, list) and len(hf_data) > 0:
        hf_data = hf_data[0]
    
    if isinstance(hf_data, dict):
        prediction = hf_data.get("symptom_result", {}).get("prediction")
        xray_results = hf_data.get("xray_result")
        return prediction, xray_results
    
    return None, None


def _format_message(prediction: str | None, xray_results: dict | None) -> str:
    """
    Format a clean message with prediction and confidence.
    
    Args:
        prediction: Disease prediction string
        xray_results: X-ray confidence scores
        
    Returns:
        str: Formatted message with prediction and confidence
    """
    if not prediction:
        return "Unable to determine prediction"
    
    # Get max confidence from xray results
    max_confidence = 0.0
    if xray_results and isinstance(xray_results, dict):
        max_confidence = max(xray_results.values()) if xray_results else 0.0
    
    confidence_pct = round(max_confidence * 100, 1)
    return f"Predicted: {prediction}\nConfidence: {confidence_pct}%"


@router.post("/predict", response_model=PredictResponse, summary="Get ML model prediction from HF Space")
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Get model prediction from Hugging Face Space.
    
    Sends user input to the HF Space ML model and returns the prediction result.
    
    Args:
        payload: Request containing symptom input text
        
    Returns:
        dict: Contains status, prediction (disease name), and xray_results separately
        
    Raises:
        HTTPException: If HF Space API fails
    """
    try:
        _LOGGER.info(f"[ML Endpoint] Prediction request received. Input length: {len(payload.input)}")
        _LOGGER.info(f"[ML Endpoint] Calling HF Space...")
        
        hf_response = await call_hf_predict(payload.input)
        
        _LOGGER.info(f"[ML Endpoint] HF Space returned successfully. Response type: {type(hf_response)}")
        
        # Parse and separate prediction from xray results
        prediction, xray_results = _parse_hf_response(hf_response)
        
        # Calculate confidence from xray results
        confidence = 0.0
        if xray_results and isinstance(xray_results, dict):
            scores = [v for v in xray_results.values() if isinstance(v, (int, float))]
            if scores:
                confidence = max(scores)
        
        # Return structured response
        return PredictResponse(
            status="success",
            prediction=prediction,
            confidence=confidence,
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

