"""Disease prediction API endpoints."""

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from app.services.disease_prediction_service import (
    get_disease_prediction_service,
    DiseasePredictionService,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SymptomListResponse(BaseModel):
    """Response model for symptoms list."""
    symptoms: list[str] = Field(description="List of all available symptoms")
    count: int = Field(description="Number of symptoms available")


class DiseasePredictionRequest(BaseModel):
    """Request model for disease prediction."""
    symptoms: list[str] = Field(
        ...,
        min_items=1,
        description="List of symptoms to analyze"
    )


class DiseasePredictionResponse(BaseModel):
    """Response model for disease prediction."""
    predicted_disease: str = Field(description="Predicted disease name")
    confidence: float = Field(description="Confidence level (0-100)")
    matched_symptoms: list[str] = Field(description="Symptoms that were matched")
    input_count: int = Field(description="Number of input symptoms")
    valid_count: int = Field(description="Number of valid symptoms")


@router.get("/symptoms", response_model=SymptomListResponse)
async def get_symptoms(
    service: DiseasePredictionService = Depends(get_disease_prediction_service)
) -> SymptomListResponse:
    """
    Get the list of all available symptoms.
    
    Returns a list of 377 symptoms that can be used for disease prediction.
    """
    try:
        if not service.symptoms_list:
            logger.warning("No symptoms loaded in service")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Disease prediction models are not available. Please contact administrator.",
            )
        
        return SymptomListResponse(
            symptoms=service.symptoms_list,
            count=len(service.symptoms_list)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error getting symptoms list: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve symptoms list.",
        ) from exc


@router.post("/predict-disease", response_model=DiseasePredictionResponse)
async def predict_disease(
    request: DiseasePredictionRequest,
    service: DiseasePredictionService = Depends(get_disease_prediction_service),
) -> DiseasePredictionResponse:
    """
    Predict disease based on provided symptoms.
    
    Takes a list of symptom names and returns the predicted disease with confidence.
    
    Args:
        request: Contains list of symptoms to analyze
        
    Returns:
        Predicted disease name and confidence level
    """
    try:
        if not service.is_ready:
            logger.warning("Disease prediction service not ready")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Disease prediction models are not available. Please contact administrator.",
            )
        
        if not request.symptoms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one symptom is required for prediction.",
            )
        
        # Remove duplicates and empty strings
        symptoms = [s.strip() for s in request.symptoms if s.strip()]
        symptoms = list(set(symptoms))  # Remove duplicates
        
        if not symptoms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid symptoms provided.",
            )
        
        prediction = service.predict_disease(symptoms)
        
        return DiseasePredictionResponse(**prediction)
        
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(f"Validation error during prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"Error during disease prediction: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Disease prediction failed. Please try again.",
        ) from exc
