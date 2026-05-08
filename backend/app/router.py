"""Primary inference routing: chooses MedGemma vs Gemini before the orchestrator runs."""

from __future__ import annotations

import re
from enum import Enum


class InferencePrimary(str, Enum):
    MEDGEMMA = "medgemma"
    GEMINI = "gemini"


_COMPLEX_MEDICAL_MARKERS = (
    "differential diagnosis",
    "pathophysiology",
    "mechanism of action",
    "correlate",
    "correlation",
    "interpret these",
    "interpret this",
    "critique",
    "critically appraise",
    "icu",
    "sepsis pathway",
    "acs stemi",
    "stemi",
    "nccn",
    "staging",
    "prognosis multifactor",
    "multimodal imaging",
)


def estimate_file_evidence_chars(file_contexts: list[dict] | None) -> int:
    if not file_contexts:
        return 0
    total = 0
    for item in file_contexts:
        t = item.get("extracted_text") or ""
        total += len(t)
    return total


def _is_complex_medical_reasoning(text: str) -> bool:
    if not text or not text.strip():
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _COMPLEX_MEDICAL_MARKERS):
        return True
    if len(text) > 1200:
        # Long clinical narratives / pasted notes often benefit from local multimodal reasoning.
        return any(ch.isalpha() for ch in text) and any(
            w in lowered
            for w in (
                "patient",
                "diagnosis",
                "symptom",
                "treatment",
                "medic",
                "clinic",
                "lab",
                "imaging",
                "ct",
                "mri",
            )
        )
    return False


def _normalize_question_contractions(text: str) -> str:
    """Map what's / whats → what is for lightweight definitional routing."""
    t = text.lower().strip()
    t = re.sub(r"\bwhat'?s\b", "what is", t)
    t = re.sub(r"\bwhat are\b", "what are", t)
    return t


def _is_app_or_system_question(text: str) -> bool:
    """Detect if query is about the app/system itself, not medical."""
    app_keywords = (
        "ui",
        "ux",
        "navigation",
        "feature",
        "button",
        "page",
        "route",
        "login",
        "register",
        "account",
        "profile",
        "dashboard",
        "error",
        "bug",
        "crash",
        "technical",
        "screen",
        "app",
        "application",
        "website",
        "frontend",
        "backend",
        "api",
        "how do i use",
        "how to use",
        "how do i access",
        "where is the",
        "how do i find",
    )
    lowered = text.lower()
    return any(keyword in lowered for keyword in app_keywords)


def _looks_conversational_or_light_medical(text: str) -> bool:
    lowered = text.lower().strip()
    if len(lowered) >= 400:
        return False
    greeting_ok = lowered in {"hi", "hello", "hey", "yo", "thanks", "thank you"} or lowered.startswith(
        ("hello ", "hi ", "hey ", "thanks ", "thank you ")
    )
    if greeting_ok:
        return True
    norm = _normalize_question_contractions(text)
    return any(fragment in norm for fragment in ("define ", "what is ", "what are "))


def decide_primary_route(
    *,
    has_image: bool,
    text: str,
    total_file_chars: int = 0,
) -> InferencePrimary:
    """
    MEDGEMMA-FIRST routing strategy (user spec):
    - ALWAYS route to MedGemma for medical/clinical queries unless proven to be app-related
    - Use Gemini ONLY for:
      1. Application/UI/technical questions
      2. As fallback if MedGemma fails/has low confidence
    
    Rules:
    - Image → MedGemma (vision)
    - Complex medical reasoning → MedGemma
    - Substantial uploaded evidence (≥4000 chars) → MedGemma
    - App/system question (UI, navigation, account) → Gemini
    - Otherwise → MedGemma (assume medical unless proven otherwise)
    """
    # Vision always goes to MedGemma
    if has_image:
        return InferencePrimary.MEDGEMMA

    # Complex medical or substantial evidence → MedGemma
    if _is_complex_medical_reasoning(text) or total_file_chars >= 4000:
        return InferencePrimary.MEDGEMMA

    # App/system question → Gemini
    if _is_app_or_system_question(text):
        return InferencePrimary.GEMINI

    # Default: assume medical and route to MedGemma
    # (Orchestrator will fallback to Gemini if MedGemma fails/low confidence)
    return InferencePrimary.MEDGEMMA
