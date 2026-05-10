"""Disease prediction API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.disease_prediction_service import get_disease_prediction_service
from app.services.hf_client import call_hf_predict, HFClientError

_LOGGER = logging.getLogger(__name__)

router = APIRouter()

_disease_prediction_service = get_disease_prediction_service()


class DiseasePredictor(BaseModel):
    """Request schema for disease prediction."""
    symptoms: str = Field(..., min_length=1, max_length=5000, description="Symptom description")


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
        dict: HF Space model prediction response
    """
    try:
        _LOGGER.info(f"[Disease Prediction] Predicting from symptoms. Length: {len(payload.symptoms)}")
        
        result = await call_hf_predict(payload.symptoms)
        
        _LOGGER.info(f"[Disease Prediction] HF Space returned successfully")
        
        return {
            "status": "success",
            "prediction": result
        }
        
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
