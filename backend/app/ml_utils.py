"""
Lightweight ML utilities for the backend.

This module intentionally avoids importing heavy ML libraries (torch, transformers,
sklearn, etc.). It provides small helpers to detect model files and return a
safe proxy when models are not available so the API can continue running.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def model_file_candidates(base_name: str) -> List[Path]:
    """Return possible file locations for a given model base name."""
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "models" / f"{base_name}.pkl",
        repo_root / "storage" / f"{base_name}.pkl",
        repo_root / f"{base_name}.pkl",
    ]


def model_exists(base_name: str) -> bool:
    """Return True if any candidate model file exists."""
    for p in model_file_candidates(base_name):
        if p.exists():
            return True
    return False


class ModelUnavailableError(RuntimeError):
    pass


class ModelProxy:
    """A safe proxy for ML models when real model loading is disabled.

    Methods mimic common model APIs (`predict`, `predict_proba`, etc.) but
    raise `ModelUnavailableError` with a standardized message.
    """

    def __init__(self, name: str | None = None):
        self.name = name or "unknown"

    def _unavailable(self, *args, **kwargs):
        raise ModelUnavailableError("model_service_not_available")

    def predict(self, *args, **kwargs):
        return self._unavailable()

    def predict_proba(self, *args, **kwargs):
        return self._unavailable()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<ModelProxy name={self.name}>"
