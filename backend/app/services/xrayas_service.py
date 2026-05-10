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
        Analyze X-ray image using HF Space model via predict endpoint.
        
        Args:
            image_path: Path to X-ray image file
            text: Optional text description/notes
            image_bytes: Actual image file bytes
            
        Returns:
            dict: Analysis result from HF Space
        """
        _LOGGER.info("[XrayAS] Analyzing image: %s", image_path)

        if image_bytes is None or len(image_bytes) == 0:
            _LOGGER.error("[XrayAS] Image bytes are empty or None")
            raise ValueError("Image bytes are required for X-ray analysis")
        
        _LOGGER.info(f"[XrayAS] Image size: {len(image_bytes)} bytes")

        try:
            # Encode image as base64 and send to predict endpoint
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create input text with image and optional notes
            input_text = f"xray_analysis image_base64={encoded_image[:10000]}"
            if text:
                input_text += f" notes={text.strip()}"
            
            _LOGGER.info("[XrayAS] Sending to predict endpoint with image")
            result = await call_hf_predict(input_text)
            
            # Parse result
            answer = str(result)
            confidence = 0.82
            
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            if isinstance(result, dict):
                if "xray_result" in result:
                    # Extract from xray_result confidence scores
                    xray_data = result["xray_result"]
                    if isinstance(xray_data, dict):
                        # Get findings from dict values
                        scores = [v for v in xray_data.values() if isinstance(v, (int, float))]
                        if scores:
                            confidence = max(scores)
                            # Format top findings
                            top_findings = sorted(xray_data.items(), key=lambda x: x[1], reverse=True)[:3]
                            answer = f"X-ray findings: {', '.join([f'{k}: {round(v*100,1)}%' for k,v in top_findings])}"
                elif "symptom_result" in result:
                    answer = result["symptom_result"].get("prediction", "Unable to analyze image")
            
            return {
                "ok": True,
                "answer": answer,
                "confidence": confidence,
                "model_name": "HF Space",
                "raw": result,
            }
            
        except HFClientError as exc:
            _LOGGER.error(f"[XrayAS] HF Client error: {exc}")
            raise
        except Exception as exc:
            _LOGGER.error(f"[XrayAS] Error analyzing image: {exc}", exc_info=True)
            raise HFClientError(f"X-ray analysis failed: {exc}") from exc


# Singleton instance
xrayas_service_instance = XrayASService()
