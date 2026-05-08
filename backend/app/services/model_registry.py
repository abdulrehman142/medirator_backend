"""
Model Registry: Singleton for loading and managing ML model artifacts.
"""

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for ML model artifacts.
    Holds family_model, symptom_model, label_encoder, and symptoms_list.
    """

    def __init__(self):
        self.family_model: Any = None
        self.symptom_model: Any = None
        self.label_encoder: Any = None
        self.symptoms_list: list[str] = []
        self.available: bool = False
        self.load_error: str | None = None

    def load(self, base_dir: Path) -> None:
        """
        Load all model artifacts from disk.

        Args:
            base_dir: Base directory where model files are stored

        Raises:
            FileNotFoundError: If any model file is missing
            pickle.UnpicklingError: If any file cannot be unpickled
            Exception: For any other loading error
        """
        try:
            base_path = Path(base_dir)

            # Define expected model file paths (directly in backend folder)
            family_model_path = base_path / "family_model.pkl"
            symptom_model_path = base_path / "symptom_model.pkl"
            label_encoder_path = base_path / "label_encoder.pkl"
            symptoms_list_path = base_path / "symptoms_list.pkl"

            # Load family model
            logger.info(f"Loading family model from {family_model_path}")
            self.family_model = joblib.load(family_model_path)
            logger.info("Family model loaded successfully")

            # Load symptom model
            logger.info(f"Loading symptom model from {symptom_model_path}")
            self.symptom_model = joblib.load(symptom_model_path)
            logger.info("Symptom model loaded successfully")

            # Load label encoder
            logger.info(f"Loading label encoder from {label_encoder_path}")
            self.label_encoder = joblib.load(label_encoder_path)
            logger.info("Label encoder loaded successfully")

            # Load symptoms list
            logger.info(f"Loading symptoms list from {symptoms_list_path}")
            self.symptoms_list = joblib.load(symptoms_list_path)
            logger.info("Symptoms list loaded successfully")

            self.available = True
            self.load_error = None
            logger.info("All model artifacts loaded successfully")

        except FileNotFoundError as e:
            self.available = False
            self.load_error = f"Model file not found: {str(e)}"
            logger.error(f"Model loading failed: {self.load_error}")
            raise

        except Exception as e:
            self.available = False
            self.load_error = f"Unexpected error loading models: {str(e)}"
            logger.error(f"Model loading failed: {self.load_error}")
            raise

    def is_available(self) -> bool:
        """Check if all models are available."""
        return self.available


# Module-level singleton instance
model_registry = ModelRegistry()
