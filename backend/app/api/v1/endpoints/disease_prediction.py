"""Disease prediction API endpoints."""

from fastapi import APIRouter

from app.services.disease_prediction_service import get_disease_prediction_service

router = APIRouter()

_disease_prediction_service = get_disease_prediction_service()


@router.get('/symptoms')
async def get_symptoms():
    return {"symptoms": _disease_prediction_service.get_symptoms()}


@router.post('/predict-disease')
async def predict_disease():
    return {"message": "model_service_not_available"}
