"""Disease prediction service using ML models."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "models" / "symptom_model.pkl",
    Path(__file__).resolve().parents[2] / "storage" / "symptom_model.pkl",
]

_SYMPTOMS_LIST_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "models" / "symptoms_list.pkl",
    Path(__file__).resolve().parents[2] / "storage" / "symptoms_list.pkl",
]

_LABEL_ENCODER_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "models" / "label_encoder.pkl",
    Path(__file__).resolve().parents[2] / "storage" / "label_encoder.pkl",
]


@lru_cache(maxsize=1)
def load_symptom_model() -> Any:
    """Load the symptom model once per process."""
    for model_path in _MODEL_CANDIDATES:
        if model_path.exists():
            logger.info(f"Loading symptom model from {model_path}")
            return joblib.load(model_path)
    
    logger.warning(f"Symptom model not found at: {_MODEL_CANDIDATES}")
    return None


@lru_cache(maxsize=1)
def load_symptoms_list() -> list[str]:
    """Load the symptoms list once per process."""
    for symptoms_path in _SYMPTOMS_LIST_CANDIDATES:
        if symptoms_path.exists():
            logger.info(f"Loading symptoms list from {symptoms_path}")
            symptoms = joblib.load(symptoms_path)
            if isinstance(symptoms, list):
                return symptoms
            return list(symptoms)
    
    logger.warning(f"Symptoms list not found at: {_SYMPTOMS_LIST_CANDIDATES}")
    # Return empty list if not found; API will handle gracefully
    return []


@lru_cache(maxsize=1)
def load_label_encoder() -> Any:
    """Load the label encoder once per process."""
    for encoder_path in _LABEL_ENCODER_CANDIDATES:
        if encoder_path.exists():
            logger.info(f"Loading label encoder from {encoder_path}")
            return joblib.load(encoder_path)
    
    logger.warning(f"Label encoder not found at: {_LABEL_ENCODER_CANDIDATES}")
    return None


class DiseasePredictionService:
    """Service for predicting diseases based on symptoms."""

    def __init__(self):
        """Initialize the service with loaded models."""
        self.model = load_symptom_model()
        self.symptoms_list = load_symptoms_list()
        self.label_encoder = load_label_encoder()
        self.is_ready = self.model is not None and self.symptoms_list and self.label_encoder is not None

    def get_symptoms(self) -> list[str]:
        """Return the list of all available symptoms."""
        return self.symptoms_list

    def predict_disease(self, input_symptoms: list[str]) -> dict[str, Any]:
        """
        Predict disease based on input symptoms.
        
        Args:
            input_symptoms: List of symptom names
            
        Returns:
            Dictionary with predicted disease and confidence
        """
        if not self.is_ready:
            raise ValueError(
                "Disease prediction models not loaded. "
                "Please ensure symptom_model.pkl, symptoms_list.pkl, and label_encoder.pkl are available."
            )

        # Create feature vector: 377 length with 1s for selected symptoms
        feature_vector = self._create_feature_vector(input_symptoms)
        
        if feature_vector is None:
            return {
                "predicted_disease": "Unable to process",
                "confidence": 0.0,
                "matched_symptoms": [],
                "error": "No valid symptoms provided"
            }

        # Get prediction from model
        try:
            prediction = self.model.predict([feature_vector])[0]
            
            # Get confidence if model supports predict_proba
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities = self.model.predict_proba([feature_vector])[0]
                    confidence = float(np.max(probabilities)) * 100.0
                except Exception as e:
                    logger.warning(f"Could not get confidence: {e}")
            
            # Decode prediction to disease name.
            # Some trained models emit the class label directly, while others emit an encoded index.
            if isinstance(prediction, (int, np.integer)):
                disease_name = self.label_encoder.inverse_transform([prediction])[0]
            else:
                disease_name = str(prediction)
            
            # Get matched symptoms
            matched_symptoms = [
                sym for sym in input_symptoms 
                if sym.lower() in [s.lower() for s in self.symptoms_list]
            ]
            
            return {
                "predicted_disease": disease_name,
                "confidence": min(confidence, 100.0),
                "matched_symptoms": matched_symptoms,
                "input_count": len(input_symptoms),
                "valid_count": len(matched_symptoms),
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ValueError(f"Model prediction failed: {str(e)}")

    def _create_feature_vector(self, input_symptoms: list[str]) -> np.ndarray | None:
        """
        Create a 377-length feature vector from input symptoms.
        
        Args:
            input_symptoms: List of symptom names
            
        Returns:
            Feature vector as numpy array, or None if no valid symptoms
        """
        # Initialize zero vector
        vector = np.zeros(len(self.symptoms_list), dtype=np.int32)
        
        # Normalize symptoms list (lowercase for matching)
        normalized_symptoms = {s.lower(): i for i, s in enumerate(self.symptoms_list)}
        
        # Set 1 for each matched symptom
        matched_count = 0
        for symptom in input_symptoms:
            normalized = symptom.lower().strip()
            if normalized in normalized_symptoms:
                idx = normalized_symptoms[normalized]
                vector[idx] = 1
                matched_count += 1
        
        # Return None if no symptoms matched
        return vector if matched_count > 0 else None


@lru_cache(maxsize=1)
def get_disease_prediction_service() -> DiseasePredictionService:
    """Get the cached disease prediction service."""
    return DiseasePredictionService()
