"""Hugging Face Space API client for ML model predictions."""

import httpx
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Hugging Face Space API endpoint
HF_SPACE_URL = "https://AbdulRehman142-medirator_mlapi.hf.space/run/predict"
HF_REQUEST_TIMEOUT = 60.0  # 60 seconds timeout


class HFClientError(Exception):
    """Custom exception for HF Space API errors."""
    pass


async def call_hf_predict(user_input: str) -> dict[str, Any]:
    """
    Call the Hugging Face Space predict endpoint with user input.
    
    Args:
        user_input: Symptom text or input to send to the model
        
    Returns:
        dict: Response from HF Space containing model prediction
        
    Raises:
        HFClientError: If the API call fails
    """
    try:
        payload = {
            "data": [user_input]
        }
        
        async with httpx.AsyncClient(timeout=HF_REQUEST_TIMEOUT) as client:
            response = await client.post(HF_SPACE_URL, json=payload)
        
        # Check for successful response
        if response.status_code != 200:
            error_detail = response.text or f"HTTP {response.status_code}"
            _LOGGER.error(f"HF Space API error: {error_detail}")
            raise HFClientError(f"HF Space returned {response.status_code}: {error_detail}")
        
        # Parse and return JSON response
        result = response.json()
        _LOGGER.info(f"HF Space prediction successful for input length: {len(user_input)}")
        return result
        
    except httpx.TimeoutException as exc:
        _LOGGER.error(f"HF Space API timeout after {HF_REQUEST_TIMEOUT}s")
        raise HFClientError("HF Space model service timeout") from exc
        
    except httpx.ConnectError as exc:
        _LOGGER.error(f"HF Space connection error: {exc}")
        raise HFClientError("Failed to connect to HF Space model service") from exc
        
    except httpx.HTTPError as exc:
        _LOGGER.error(f"HF Space HTTP error: {exc}")
        raise HFClientError("HF Space model service error") from exc
        
    except ValueError as exc:
        _LOGGER.error(f"HF Space response is not valid JSON: {exc}")
        raise HFClientError("Invalid response from HF Space model service") from exc
        
    except Exception as exc:
        _LOGGER.error(f"Unexpected error calling HF Space: {exc}")
        raise HFClientError("Unexpected error from HF Space model service") from exc
