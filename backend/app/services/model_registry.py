"""
Model Registry: Singleton for loading and managing ML model artifacts.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Placeholder model registry for deployments without ML.

    This registry does not attempt to load model artifacts. It preserves the
    API used by the rest of the backend, but always reports models as
    unavailable. The real model registry (with joblib loading) was moved to
    the `ML/` folder for offline work.
    """

    def __init__(self):
        self.family_model: Any = None
        self.symptom_model: Any = None
        self.label_encoder: Any = None
        self.symptoms_list: list[str] = []
        self.available: bool = False
        self.load_error: str | None = "model_service_not_available"

    def load(self, base_dir: Path) -> None:  # pragma: no cover - placeholder
        self.available = False
        self.load_error = "model_service_not_available"

    def is_available(self) -> bool:
        return False


# Module-level singleton instance
model_registry = ModelRegistry()
