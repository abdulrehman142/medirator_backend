"""
Lightweight ML service interface used by backend routes.

This module intentionally avoids importing heavy ML libraries. It exposes a
`get_model_proxy` helper and a small `MLService` class that routes can depend
on. When real models are not available, it returns `ModelProxy` instances that
raise a standardized `ModelUnavailableError` with message
`"model_service_not_available"`.
"""
from __future__ import annotations

from typing import Any, Optional
from app.ml_utils import ModelProxy, model_exists


def get_model_proxy(name: str) -> ModelProxy:
    """Return a ModelProxy if no real model is present.

    If in future you want to enable local inference you can extend this
    function to load models conditionally (while keeping heavy imports
    behind try/except or optional extras).
    """
    if model_exists(name):
        # Conservative behavior: do not attempt to load heavy models in the
        # Render deployment path. Return proxy to keep API stable.
        return ModelProxy(name)
    return ModelProxy(name)


class MLService:
    """Simple service wrapper for routes to depend on.

    Methods return proxies or standardized placeholder responses so the
    rest of the backend can continue to operate without ML dependencies.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = get_model_proxy(model_name)

    def is_ready(self) -> bool:
        # Always False for proxy; replace with real check when enabling ML.
        return False

    def predict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "model_service_not_available"}
