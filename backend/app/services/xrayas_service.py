"""
XRayAS service for backend deployment via Hugging Face Space.

Uses HF Space API for chest X-ray analysis predictions.
"""

from __future__ import annotations

import logging
import base64
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

    async def analyze(
        self,
        image_path: str,
        text: str | None = None,
        image_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        """
        Analyze X-ray image using HF Space model.
        
        Args:
            image_path: Path to X-ray image file
            text: Optional text description
            
        Returns:
            dict: Analysis result from HF Space
        """
        _LOGGER.info("[XrayAS] Analyzing image: %s", image_path)

        input_text = f"xray_analysis image={image_path}"
        if text:
            input_text = f"{input_text} notes={text.strip()}"
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            input_text = f"{input_text} image_base64={encoded[:4000]}"

        try:
            result = await call_hf_predict(input_text)
        except HFClientError:
            raise

        answer = str(result)
        confidence = 0.0

        if isinstance(result, dict):
            data = result.get("data")
            if isinstance(data, list) and data:
                answer = str(data[0])
                for item in data:
                    if isinstance(item, (float, int)):
                        confidence = float(item)
                        break
            answer = str(result.get("answer") or answer)
            confidence = float(result.get("confidence", confidence) or confidence)

        return {
            "ok": True,
            "answer": answer,
            "confidence": confidence,
            "model_name": "HF Space",
            "raw": result,
        }


xrayas_service_instance = XrayASService()
