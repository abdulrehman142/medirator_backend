"""Lightweight symptom catalog used by the backend symptom endpoint.

This keeps the backend free of heavy model artifacts while still allowing the
`/symptom-predictor/symptoms` route to return a usable list in local dev.
"""

DEFAULT_SYMPTOMS: list[str] = [
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


def get_default_symptoms() -> list[str]:
    return list(DEFAULT_SYMPTOMS)
