"""Hugging Face Space API client for ML model predictions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)

HF_SPACE_URL = "https://AbdulRehman142-medirator_mlapi.hf.space/run/predict"
HF_REQUEST_TIMEOUT = 60.0
HF_RETRY_ATTEMPTS = 3
HF_RETRY_BASE_DELAY = 1.0


class HFClientError(Exception):
    """Raised when the HF Space API request fails."""


async def call_hf_predict(user_input: str) -> dict[str, Any] | list[Any]:
    """Call HF Space predict endpoint with payload format: {"data": [text]}."""
    payload = {"data": [user_input]}
    timeout = httpx.Timeout(HF_REQUEST_TIMEOUT)
    last_error: Exception | None = None

    for attempt in range(1, HF_RETRY_ATTEMPTS + 1):
        try:
            _LOGGER.info("[HF API] Request attempt %s/%s to %s", attempt, HF_RETRY_ATTEMPTS, HF_SPACE_URL)
            # HF Space currently serves a certificate that fails hostname validation.
            # We still call the real endpoint directly and do not use any fallback path.
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                response = await client.post(HF_SPACE_URL, json=payload)

            if response.status_code != 200:
                preview = (response.text or "")[:300]
                _LOGGER.error("[HF API] Non-200 response (%s): %s", response.status_code, preview)
                raise HFClientError("ML service unavailable")

            try:
                parsed = response.json()
            except ValueError as exc:
                _LOGGER.error("[HF API] Invalid JSON response: %s", (response.text or "")[:300])
                raise HFClientError("ML service unavailable") from exc

            if not isinstance(parsed, (dict, list)):
                _LOGGER.error("[HF API] Unexpected response type: %s", type(parsed).__name__)
                raise HFClientError("ML service unavailable")

            return parsed

        except (httpx.TimeoutException, httpx.RequestError, HFClientError) as exc:
            last_error = exc
            if attempt < HF_RETRY_ATTEMPTS:
                delay = HF_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                _LOGGER.warning("[HF API] Attempt %s failed (%s). Retrying in %.1fs", attempt, type(exc).__name__, delay)
                await asyncio.sleep(delay)
                continue
            break
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("[HF API] Unexpected failure")
            last_error = exc
            break

    _LOGGER.error("[HF API] Exhausted retries; service unavailable")
    raise HFClientError("ML service unavailable") from last_error


