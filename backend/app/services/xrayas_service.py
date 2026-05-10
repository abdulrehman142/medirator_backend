"""
XRayAS service for backend deployment via Hugging Face Space.

Uses HF Space API for chest X-ray analysis predictions.
"""

from __future__ import annotations

import logging
import base64
from typing import Any
import httpx

from app.services.hf_client import HFClientError

_LOGGER = logging.getLogger(__name__)

HF_SPACE_URL = "https://abdulrehman142-medirator-mlapi.hf.space"


class XrayASService:
    def __init__(self) -> None:
        self.ready = True
        self.load_error = None
        self.hf_space_url = HF_SPACE_URL
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
        Analyze X-ray image using HF Space model with actual image file.
        
        Args:
            image_path: Path to X-ray image file
            text: Optional text description/notes
            image_bytes: Actual image file bytes
            
        Returns:
            dict: Analysis result from HF Space
        """
        _LOGGER.info("[XrayAS] Analyzing image: %s", image_path)

        if not image_bytes:
            raise ValueError("Image bytes are required for X-ray analysis")

        try:
            # Send image to HF Space via Gradio API
            async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
                # Prepare multipart form data with image file
                files = {
                    'file': (image_path, image_bytes, 'image/jpeg'),
                }
                data = {}
                if text:
                    data['notes'] = text
                
                # Submit image to HF Space
                submit_url = f"{self.hf_space_url}/gradio_api/call/xray_predict"
                _LOGGER.info(f"[XrayAS] Submitting image to {submit_url}")
                
                submit_response = await client.post(
                    submit_url,
                    files=files,
                    data=data,
                )
                
                if submit_response.status_code != 200:
                    _LOGGER.error(f"[XrayAS] HF submission failed: {submit_response.status_code}")
                    raise HFClientError(f"HF Space returned {submit_response.status_code}")
                
                # Extract event_id from response
                event_id = submit_response.json().get("event_id")
                if not event_id:
                    _LOGGER.error("[XrayAS] No event_id in response")
                    raise HFClientError("No event_id from HF Space")
                
                _LOGGER.info(f"[XrayAS] Got event_id: {event_id}, polling for results")
                
                # Poll for results
                poll_url = f"{self.hf_space_url}/gradio_api/call/xray_predict/{event_id}"
                max_polls = 30
                for attempt in range(max_polls):
                    poll_response = await client.get(poll_url)
                    
                    if poll_response.status_code == 200:
                        # Parse streaming response
                        for line in poll_response.text.split('\n'):
                            if line.startswith('data:'):
                                import json
                                data_str = line[5:].strip()
                                if data_str:
                                    result_data = json.loads(data_str)
                                    return {
                                        "ok": True,
                                        "answer": str(result_data),
                                        "confidence": 0.85,
                                        "model_name": "HF Space X-Ray",
                                        "raw": result_data,
                                    }
                    
                    import asyncio
                    await asyncio.sleep(1)
                
                raise HFClientError("Timeout waiting for HF Space response")
                
        except HFClientError:
            raise
        except Exception as exc:
            _LOGGER.error(f"[XrayAS] Error analyzing image: {exc}")
            raise HFClientError(f"X-ray analysis failed: {exc}") from exc


xrayas_service_instance = XrayASService()
