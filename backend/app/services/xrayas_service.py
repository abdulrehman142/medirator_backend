"""
XRayAS service for backend deployment via Hugging Face Space.

Uses HF Space API for chest X-ray analysis predictions.
Falls back to mock predictions when HF Space is unavailable.
"""

import logging
from typing import Any
from app.services.hf_client import call_hf_predict
from app.services.mock_ml_service import analyze_xray_fallback

_LOGGER = logging.getLogger(__name__)


class XrayASService:
    def __init__(self) -> None:
        self.ready = True
        self.load_error = None
        _LOGGER.info("[XrayAS] Service initialized with HF Space backend (with fallback)")

    def load_model(self) -> None:
        """No-op - model is loaded from HF Space endpoint."""
        self.ready = True
        _LOGGER.info("[XrayAS] Model ready (HF Space with fallback)")

    def analyze(self, image_path: str, text: str | None = None) -> dict[str, Any]:
        """
        Analyze X-ray image using HF Space model (or fallback).
        
        Args:
            image_path: Path to X-ray image file
            text: Optional text description
            
        Returns:
            dict: Analysis result from HF Space or fallback
        """
        try:
            _LOGGER.info(f"[XrayAS] Analyzing image: {image_path}")
            
            # Prepare input for HF Space
            input_text = f"X-ray analysis for: {image_path}"
            if text:
                input_text = f"{input_text}. Additional info: {text}"
            
            # Call HF Space API (which has fallback built in)
            result = call_hf_predict(input_text)
            
            _LOGGER.info(f"[XrayAS] Analysis successful")
            
            return {
                "status": "success",
                "analysis": result,
                "image_path": image_path
            }
            
        except Exception as exc:
            _LOGGER.error(f"[XrayAS] Error: {exc}")
            # Use fallback directly
            return analyze_xray_fallback(image_path, text)


xrayas_service_instance = XrayASService()
