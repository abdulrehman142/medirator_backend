"""
XRayAS service for backend deployment via Hugging Face Space.

Uses HF Space API for chest X-ray analysis predictions.
"""

import logging
from typing import Any
from app.services.hf_client import call_hf_predict, HFClientError

_LOGGER = logging.getLogger(__name__)


class XrayASService:
    def __init__(self) -> None:
        self.ready = True
        self.load_error = None
        _LOGGER.info("[XrayAS] Service initialized with HF Space backend")

    def load_model(self) -> None:
        """No-op - model is loaded from HF Space endpoint."""
        self.ready = True
        _LOGGER.info("[XrayAS] Model ready (HF Space)")

    def analyze(self, image_path: str, text: str | None = None) -> dict[str, Any]:
        """
        Analyze X-ray image using HF Space model.
        
        Args:
            image_path: Path to X-ray image file
            text: Optional text description
            
        Returns:
            dict: Analysis result from HF Space
        """
        try:
            _LOGGER.info(f"[XrayAS] Analyzing image: {image_path}")
            
            # Prepare input for HF Space
            input_text = f"X-ray analysis for: {image_path}"
            if text:
                input_text = f"{input_text}. Additional info: {text}"
            
            # Call HF Space API
            result = call_hf_predict(input_text)
            
            _LOGGER.info(f"[XrayAS] Analysis successful")
            
            return {
                "status": "success",
                "analysis": result,
                "image_path": image_path
            }
            
        except HFClientError as exc:
            _LOGGER.error(f"[XrayAS] HF Client error: {exc}")
            return {
                "status": "error",
                "message": "Failed to analyze X-ray",
                "details": str(exc),
                "image_path": image_path
            }
            
        except Exception as exc:
            _LOGGER.error(f"[XrayAS] Unexpected error: {exc}")
            return {
                "status": "error",
                "message": "X-ray analysis failed",
                "details": str(exc),
                "image_path": image_path
            }


xrayas_service_instance = XrayASService()
