"""Lightweight disease prediction service for backend deployment without ML."""

from typing import Any

from app.services.hf_client import call_hf_predict
from app.services.symptom_catalog import get_default_symptoms


class DiseasePredictionService:
    """Service for symptom catalog and disease prediction calls."""

    def __init__(self) -> None:
        self.is_ready = True

    def get_symptoms(self) -> list[str]:
        return get_default_symptoms()

    async def predict_disease(self, input_symptoms: list[str]) -> dict[str, Any] | list[Any]:
        return await call_hf_predict(" ".join(input_symptoms))


def get_disease_prediction_service() -> DiseasePredictionService:
    return DiseasePredictionService()
