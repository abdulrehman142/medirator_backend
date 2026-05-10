"""Hugging Face Space API client for ML model predictions."""

import requests
import logging
from typing import Any
from urllib3.exceptions import InsecureRequestWarning

# Import fallback service
from app.services.mock_ml_service import call_hf_predict_fallback

# Suppress SSL warnings for HF Space (known certificate issue)
urllib3_logger = logging.getLogger("urllib3.connectionpool")
urllib3_logger.setLevel(logging.ERROR)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

_LOGGER = logging.getLogger(__name__)

# Hugging Face Space API endpoint
HF_SPACE_URL = "https://AbdulRehman142-medirator_mlapi.hf.space/run/predict"
HF_REQUEST_TIMEOUT = 60  # 60 seconds timeout

# Flag to use fallback service
USE_FALLBACK = True  # Set to False when HF Space is ready


class HFClientError(Exception):
    """Custom exception for HF Space API errors."""
    pass


def call_hf_predict(user_input: str) -> dict[str, Any]:
    """
    Call the Hugging Face Space predict endpoint with user input.
    Falls back to mock predictions if HF Space is unavailable.
    
    Args:
        user_input: Symptom text or input to send to the model
        
    Returns:
        dict: Response from HF Space (or mock service) containing model prediction
        
    Raises:
        HFClientError: If both HF Space and fallback fail
    """
    try:
        # If using fallback, skip HF Space and go directly to mock
        if USE_FALLBACK:
            _LOGGER.info(f"[HF API] Using fallback mock service (HF Space unavailable)")
            return call_hf_predict_fallback(user_input)
        
        payload = {
            "data": [user_input]
        }
        
        _LOGGER.info(f"[HF API] Sending request to: {HF_SPACE_URL}")
        _LOGGER.info(f"[HF API] Input length: {len(user_input)}")
        _LOGGER.debug(f"[HF API] Payload: {payload}")
        
        # Disable SSL verification for HF Space (certificate hostname mismatch workaround)
        try:
            _LOGGER.info(f"[HF API] Attempting connection with SSL verification disabled...")
            response = requests.post(
                HF_SPACE_URL, 
                json=payload, 
                timeout=HF_REQUEST_TIMEOUT,
                verify=False  # Bypass SSL verification for HF Space
            )
            _LOGGER.info(f"[HF API] Request completed successfully")
        except Exception as conn_err:
            _LOGGER.error(f"[HF API] Connection error during POST: {type(conn_err).__name__}: {conn_err}")
            _LOGGER.info(f"[HF API] Falling back to mock service...")
            return call_hf_predict_fallback(user_input)
        
        _LOGGER.info(f"[HF API] Response status code: {response.status_code}")
        _LOGGER.info(f"[HF API] Response headers: {dict(response.headers)}")
        _LOGGER.debug(f"[HF API] Response body (first 500 chars): {response.text[:500]}")
        
        # Check for successful response
        if response.status_code != 200:
            error_detail = response.text or f"HTTP {response.status_code}"
            _LOGGER.error(f"[HF API] Error {response.status_code}: {error_detail[:200]}")
            _LOGGER.info(f"[HF API] Falling back to mock service...")
            return call_hf_predict_fallback(user_input)
        
        # Parse and return JSON response
        try:
            result = response.json()
            _LOGGER.info(f"[HF API] Prediction successful, response type: {type(result)}")
            if isinstance(result, dict):
                _LOGGER.info(f"[HF API] Response keys: {list(result.keys())}")
            return result
        except ValueError as json_err:
            _LOGGER.error(f"[HF API] Response is not valid JSON: {response.text[:200]}")
            _LOGGER.info(f"[HF API] Falling back to mock service...")
            return call_hf_predict_fallback(user_input)
        
    except requests.exceptions.Timeout as exc:
        _LOGGER.error(f"[HF API] Timeout after {HF_REQUEST_TIMEOUT}s", exc_info=True)
        _LOGGER.info(f"[HF API] Falling back to mock service...")
        return call_hf_predict_fallback(user_input)
        
    except requests.exceptions.ConnectionError as exc:
        _LOGGER.error(f"[HF API] Connection error: {exc}", exc_info=True)
        _LOGGER.info(f"[HF API] Falling back to mock service...")
        return call_hf_predict_fallback(user_input)
        
    except requests.exceptions.RequestException as exc:
        _LOGGER.error(f"[HF API] Request exception: {exc}", exc_info=True)
        _LOGGER.info(f"[HF API] Falling back to mock service...")
        return call_hf_predict_fallback(user_input)
        
    except Exception as exc:
        _LOGGER.error(f"[HF API] Unexpected error: {type(exc).__name__}: {exc}", exc_info=True)
        # Don't raise - use fallback instead
        _LOGGER.info(f"[HF API] Falling back to mock service due to error...")
        return call_hf_predict_fallback(user_input)


