"""Symptom catalog used by the backend symptom endpoint."""

from __future__ import annotations

import pickle
from pathlib import Path


_FALLBACK_SYMPTOMS: list[str] = [
    "fever",
    "cough",
    "shortness of breath",
    "chest pain",
    "headache",
    "fatigue",
    "sore throat",
    "runny nose",
    "nasal congestion",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal pain",
    "loss of appetite",
    "loss of taste",
    "loss of smell",
    "dizziness",
    "chills",
    "body aches",
    "muscle pain",
    "joint pain",
    "back pain",
    "rash",
    "itching",
    "swelling",
    "blurred vision",
    "red eyes",
    "ear pain",
    "difficulty swallowing",
    "wheezing",
    "palpitations",
    "rapid heartbeat",
    "weakness",
    "numbness",
    "tingling",
    "anxiety",
    "depression",
    "insomnia",
    "constipation",
    "heartburn",
    "urinary pain",
    "frequent urination",
    "blood in urine",
    "weight loss",
    "weight gain",
    "night sweats",
    "loss of balance",
    "confusion",
    "seizures",
    "fainting",
]


def _load_symptoms_from_pickle() -> list[str]:
    symptoms_path = Path(__file__).resolve().parents[2] / "models" / "symptoms_list.pkl"
    try:
        with symptoms_path.open("rb") as file_handle:
            loaded = pickle.load(file_handle)
        if isinstance(loaded, list) and all(isinstance(item, str) for item in loaded):
            return loaded
    except Exception:
        pass
    return list(_FALLBACK_SYMPTOMS)


DEFAULT_SYMPTOMS: list[str] = _load_symptoms_from_pickle()


def get_default_symptoms() -> list[str]:
    return list(DEFAULT_SYMPTOMS)
