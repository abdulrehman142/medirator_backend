"""Lightweight disease prediction service for backend deployment without ML."""

from typing import Any

from app.services.symptom_catalog import get_default_symptoms


class DiseasePredictionService:
    """Lightweight service used when models are not available."""

    def __init__(self) -> None:
        self.is_ready = False

    def get_symptoms(self) -> list[str]:
        return get_default_symptoms()

    def predict_disease(self, input_symptoms: list[str]) -> dict[str, Any]:
        return {"message": "model_service_not_available"}


def get_disease_prediction_service() -> DiseasePredictionService:
    return DiseasePredictionService()
