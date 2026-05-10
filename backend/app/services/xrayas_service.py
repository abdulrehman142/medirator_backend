"""
XRayAS placeholder service for backend deployment.

Avoids importing `torch`, `pickle`, or loading model files. Keeps API
surface for callers but returns standardized placeholder responses.
"""

from typing import Any


class XrayASService:
    def __init__(self) -> None:
        self.ready = False
        self.load_error = "model_service_not_available"

    def load_model(self) -> None:
        self.ready = False

    def analyze(self, image_path: str, text: str | None = None) -> dict[str, Any]:
        return {"message": "model_service_not_available"}


xrayas_service_instance = XrayASService()
