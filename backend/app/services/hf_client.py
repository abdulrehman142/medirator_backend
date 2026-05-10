"""Hugging Face Space API client for ML model predictions."""

import requests
import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Hugging Face Space API endpoint
HF_SPACE_URL = "https://AbdulRehman142-medirator_mlapi.hf.space/run/predict"
HF_REQUEST_TIMEOUT = 60  # 60 seconds timeout


class HFClientError(Exception):
    """Custom exception for HF Space API errors."""
    pass


def call_hf_predict(user_input: str) -> dict[str, Any]:
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
        
        _LOGGER.info(f"[HF API] Sending request to: {HF_SPACE_URL}")
        _LOGGER.debug(f"[HF API] Payload: {payload}")
        
        response = requests.post(HF_SPACE_URL, json=payload, timeout=HF_REQUEST_TIMEOUT)
        
        _LOGGER.info(f"[HF API] Response status code: {response.status_code}")
        _LOGGER.debug(f"[HF API] Response body: {response.text[:500]}")
        
        # Check for successful response
        if response.status_code != 200:
            error_detail = response.text or f"HTTP {response.status_code}"
            _LOGGER.error(f"[HF API] Error {response.status_code}: {error_detail}")
            raise HFClientError(f"HF Space returned {response.status_code}: {error_detail}")
        
        # Parse and return JSON response
        try:
            result = response.json()
            _LOGGER.info(f"[HF API] Prediction successful, response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result
        except ValueError as json_err:
            _LOGGER.error(f"[HF API] Response is not valid JSON: {response.text}")
            raise HFClientError(f"Invalid JSON response from HF Space: {json_err}") from json_err
        
    except requests.exceptions.Timeout as exc:
        _LOGGER.error(f"[HF API] Timeout after {HF_REQUEST_TIMEOUT}s")
        raise HFClientError(f"HF Space timeout after {HF_REQUEST_TIMEOUT}s") from exc
        
    except requests.exceptions.ConnectionError as exc:
        _LOGGER.error(f"[HF API] Connection error: {exc}")
        raise HFClientError(f"Failed to connect to HF Space: {exc}") from exc
        
    except requests.exceptions.RequestException as exc:
        _LOGGER.error(f"[HF API] Request error: {exc}")
        raise HFClientError(f"HF Space request error: {exc}") from exc
        
    except Exception as exc:
        _LOGGER.error(f"[HF API] Unexpected error: {type(exc).__name__}: {exc}")
        raise HFClientError(f"Unexpected error calling HF Space: {exc}") from exc

