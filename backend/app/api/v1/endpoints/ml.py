"""ML model prediction endpoints via Hugging Face Space."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.hf_client import call_hf_predict, HFClientError

router = APIRouter()


class PredictRequest(BaseModel):
    """Request schema for ML prediction."""
    input: str = Field(..., min_length=1, max_length=5000, description="User symptom input text")


class PredictResponse(BaseModel):
    """Response schema for ML prediction."""
    status: str
    data: dict | list | str | None = None
    message: str | None = None


@router.post("/predict", response_model=PredictResponse, summary="Get ML model prediction")
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
        # Call HF Space API
        hf_response = await call_hf_predict(payload.input)
        
        # Return successful response with HF data
        return PredictResponse(
            status="success",
            data=hf_response,
            message=None
        )
        
    except HFClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model service temporarily unavailable"
        ) from exc
        
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process prediction request"
        ) from exc
