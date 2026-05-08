"""Google Gemini client wrapper (API key loaded from Settings / environment only)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from google import genai
from google.genai import errors as genai_errors

from app.core.config import Settings

_LOGGER = logging.getLogger(__name__)

# Short model IDs only (SDK accepts bare slugs — avoid duplicate calls with `models/...`).
# Order: stabler quotas first — 2.5 can return 429 on free/low tiers faster than 2.0-flash.
_FALLBACK_MODEL_IDS: tuple[str, ...] = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)


class GeminiService:
    """Thin façade around `google.genai` with resilient model fallback."""

    def __init__(self, settings: Settings):
        # Allow disabling Gemini via settings (GEMINI_ENABLED=false)
        enabled = getattr(settings, "gemini_enabled", True)
        if not enabled:
            self.client = None
            self._model_cache = None
            return

        raw = getattr(settings, "gemini_api_key", None)
        api_key = raw.strip() if isinstance(raw, str) else raw
        api_key = api_key if api_key else None
        if api_key:
            self.client: genai.Client | None = genai.Client(api_key=api_key)
        else:
            self.client = None
        self._model_cache: Optional[str] = None

    def configured(self) -> bool:
        return self.client is not None

    def _register_working_slug(self, model_id: str) -> None:
        if model_id.startswith("models/"):
            self._model_cache = model_id.removeprefix("models/")
        else:
            self._model_cache = model_id

    def get_supported_model(self) -> Optional[str]:
        """Best-effort discovery via list_models (optional; callers still try fallbacks if this is None)."""
        if not self.client:
            return None
        if self._model_cache:
            return self._model_cache

        preferred = (
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-2.0-flash",
            "models/gemini-2.0-pro",
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
        )

        try:
            pager = self.client.models.list()
            available = [m.name for m in pager if getattr(m, "name", None)]
            if not available:
                return None

            available_set = set(available)

            def pick_short_from(name: str) -> str | None:
                if "/models/" in name:
                    return name.split("/models/", 1)[-1].split(":")[0]
                if name.startswith("models/"):
                    return name.removeprefix("models/")
                return name if "/" not in name else None

            for candidate in preferred:
                # Exact match common on AI Studio backends
                if candidate in available_set:
                    slug = candidate.removeprefix("models/")
                    self._register_working_slug(slug)
                    return slug

            for fallback in preferred:
                tail = fallback.removeprefix("models/")
                for av in available:
                    if fallback in av or tail in av or av.endswith(tail):
                        slug = pick_short_from(av)
                        if slug:
                            self._register_working_slug(slug)
                            return slug

            for av in sorted(available):
                low = av.lower()
                if "gemini" in low:
                    slug = pick_short_from(av)
                    if slug:
                        self._register_working_slug(slug)
                        return slug

        except Exception:
            return None

        return None

    @staticmethod
    def extract_text(response: Any) -> str:
        if response is None:
            return ""

        try:
            t = response.text
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            try:
                s = str(parsed).strip()
                if s:
                    return s
            except Exception:
                pass

        candidates = getattr(response, "candidates", None) or []
        buffer: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if getattr(part, "thought", False):
                    continue
                chunk = getattr(part, "text", None)
                if isinstance(chunk, str) and chunk:
                    buffer.append(chunk)
        merged = "".join(buffer).strip()
        if merged:
            return merged
        return ""

    @staticmethod
    def _canonical_slug(mid: str) -> str:
        return mid.removeprefix("models/").strip()

    @staticmethod
    def _looks_rate_limit(exc: BaseException) -> bool:
        if isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429:
            return True
        low = str(exc).lower()
        return "429" in low or "resource exhausted" in low or "too many requests" in low

    @staticmethod
    def _looks_model_not_found(exc: BaseException) -> bool:
        if isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 404:
            return True
        return "404" in str(exc)

    def _expand_model_attempts(self, model_override: Optional[str]) -> list[str]:
        seen: set[str] = set()
        attempts: list[str] = []

        def add(mid: str | None) -> None:
            if not mid:
                return
            slug = self._canonical_slug(mid)
            if not slug or slug in seen:
                return
            seen.add(slug)
            attempts.append(slug)

        add(model_override)
        slug = model_override or self.get_supported_model()
        add(slug)

        for base in _FALLBACK_MODEL_IDS:
            add(base)

        return attempts

    def generate_sync(self, prompt: str, model_override: Optional[str] = None) -> tuple[str, str]:
        if not self.client:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        attempts = self._expand_model_attempts(model_override)
        last_exc: Optional[BaseException] = None

        for mid in attempts:
            for retry_idx in range(3):
                try:
                    resp = self.client.models.generate_content(model=mid, contents=prompt)
                    extracted = GeminiService.extract_text(resp)
                    if extracted:
                        self._register_working_slug(mid)
                        return extracted, mid
                    last_exc = ValueError("gemini_empty_response")
                    break
                except BaseException as e:  # noqa: BLE001
                    last_exc = e
                    if self._looks_rate_limit(e) and retry_idx < 2:
                        delay = float(2 * (retry_idx + 1))
                        _LOGGER.warning(
                            "Gemini rate limit on %s, sleeping %.1fs (attempt %s/3)",
                            mid,
                            delay,
                            retry_idx + 1,
                        )
                        time.sleep(delay)
                        continue
                    if self._looks_model_not_found(e):
                        break
                    if not self._looks_rate_limit(e):
                        break
                    break
            # Small pause before switching models to avoid hammering the API when quota pooled.
            time.sleep(0.35)

        hint = getattr(last_exc, "message", str(last_exc))
        raise RuntimeError(f"Gemini could not produce text after retries: {hint}")
