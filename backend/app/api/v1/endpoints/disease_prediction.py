"""Disease prediction API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.services.disease_prediction_service import get_disease_prediction_service
from app.services.hf_client import call_hf_predict, HFClientError

_LOGGER = logging.getLogger(__name__)

router = APIRouter()

_disease_prediction_service = get_disease_prediction_service()


class DiseasePredictor(BaseModel):
    """Request schema for disease prediction."""
    symptoms: str = Field(..., min_length=1, max_length=5000, description="Symptom description")


class DiseasePredictionResponse(BaseModel):
    """Response schema for disease prediction."""
    status: str
    prediction: str | None = None
    message: str | None = None
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
    return f"Predicted: {prediction} (Confidence: {confidence_pct}%)"


@router.get('/symptoms')
async def get_symptoms():
    return {"symptoms": _disease_prediction_service.get_symptoms()}


@router.post('/predict-disease')
async def predict_disease(payload: DiseasePredictor):
    """
    Predict disease from symptom description using HF Space model.
    
    Args:
        payload: Request containing symptom text
        
    Returns:
        dict: Prediction result with disease name and xray confidence scores separated
    """
    try:
        _LOGGER.info(f"[Disease Prediction] Predicting from symptoms. Length: {len(payload.symptoms)}")
        
        result = await call_hf_predict(payload.symptoms)
        
        _LOGGER.info(f"[Disease Prediction] HF Space returned successfully")
        
        # Parse and separate prediction from xray results
        prediction, xray_results = _parse_hf_response(result)
        
        # Format user-friendly message
        formatted_message = _format_message(prediction, xray_results)
        
        # Return formatted text response
        return PlainTextResponse(content=formatted_message)
        
    except HFClientError as exc:
        _LOGGER.error(f"[Disease Prediction] HF Client error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "message": "ML service unavailable"},
        )
        
    except Exception as exc:
        _LOGGER.error(f"[Disease Prediction] Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "message": "Disease prediction failed",
                "details": str(exc)
            }
        ) from exc
