"""Hugging Face Space API client for ML model predictions."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

_LOGGER = logging.getLogger(__name__)

HF_SPACE_BASE_URL = "https://abdulrehman142-medirator-mlapi.hf.space"
HF_SPACE_SUBMIT_PATH = "/gradio_api/call/predict"
HF_REQUEST_TIMEOUT = 60.0
HF_RETRY_ATTEMPTS = 3
HF_RETRY_BASE_DELAY = 1.0
HF_POLL_ATTEMPTS = 12
HF_POLL_INTERVAL_SECONDS = 1.0


class HFClientError(Exception):
    """Raised when the HF Space API request fails."""


def _build_submit_url() -> str:
    return f"{HF_SPACE_BASE_URL}{HF_SPACE_SUBMIT_PATH}"


def _build_poll_url(event_id: str) -> str:
    return f"{HF_SPACE_BASE_URL}{HF_SPACE_SUBMIT_PATH}/{event_id}"


def _extract_json_from_gradio_stream(raw_text: str) -> dict[str, Any] | list[Any] | None:
    lines = raw_text.splitlines()

    for line in reversed(lines):
        if not line.startswith("data:"):
            continue

        payload = line.split(":", 1)[1].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            parsed = json.loads(payload)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            continue

    return None


async def _submit_prediction(client: httpx.AsyncClient, user_input: str) -> str:
    response = await client.post(_build_submit_url(), json={"data": [user_input]})

    if response.status_code != 200:
        _LOGGER.error("[HF API] Submit failed (%s): %s", response.status_code, (response.text or "")[:300])
        raise HFClientError("ML service unavailable")

    try:
        body = response.json()
    except ValueError as exc:
        _LOGGER.error("[HF API] Submit response is not JSON: %s", (response.text or "")[:300])
        raise HFClientError("ML service unavailable") from exc

    event_id = body.get("event_id") if isinstance(body, dict) else None
    if not event_id:
        _LOGGER.error("[HF API] Submit response missing event_id: %s", body)
        raise HFClientError("ML service unavailable")

    return str(event_id)


async def _poll_prediction_result(client: httpx.AsyncClient, event_id: str) -> dict[str, Any] | list[Any]:
    poll_url = _build_poll_url(event_id)

    for _ in range(HF_POLL_ATTEMPTS):
        response = await client.get(poll_url)

        if response.status_code != 200:
            _LOGGER.error("[HF API] Poll failed (%s): %s", response.status_code, (response.text or "")[:300])
            raise HFClientError("ML service unavailable")

        parsed = _extract_json_from_gradio_stream(response.text or "")
        if parsed is not None:
            return parsed

        await asyncio.sleep(HF_POLL_INTERVAL_SECONDS)

    _LOGGER.error("[HF API] Poll timed out waiting for complete event")
    raise HFClientError("ML service unavailable")


async def call_hf_predict(user_input: str) -> dict[str, Any] | list[Any]:
    """Call HF Space predict endpoint with payload format: {"data": [text]}."""
    timeout = httpx.Timeout(HF_REQUEST_TIMEOUT)
    last_error: Exception | None = None

    for attempt in range(1, HF_RETRY_ATTEMPTS + 1):
        try:
            _LOGGER.info("[HF API] Attempt %s/%s submit->poll flow", attempt, HF_RETRY_ATTEMPTS)

            # HF Space currently serves a certificate that fails hostname validation.
            async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                event_id = await _submit_prediction(client, user_input)
                return await _poll_prediction_result(client, event_id)

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


