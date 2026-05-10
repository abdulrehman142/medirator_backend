"""Mock ML service for fallback when HF Space is unavailable."""

import logging
import random
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Sample disease predictions database
DISEASE_DATABASE = {
    "fever": ["Flu", "COVID-19", "Malaria", "Dengue", "Typhoid"],
    "cough": ["Common Cold", "Bronchitis", "Pneumonia", "Asthma", "Tuberculosis"],
    "headache": ["Migraine", "Tension Headache", "Sinusitis", "Meningitis", "Dehydration"],
    "chest pain": ["Angina", "Heart Attack", "Pulmonary Embolism", "Pneumonia", "GERD"],
    "fatigue": ["Anemia", "Depression", "Thyroid Disorder", "Chronic Fatigue", "Sleep Apnea"],
    "shortness of breath": ["Asthma", "Pneumonia", "Heart Failure", "Pulmonary Embolism", "Anxiety"],
    "nausea": ["Gastroenteritis", "Food Poisoning", "Migraines", "Pregnancy", "Medication Side Effect"],
    "dizziness": ["Vertigo", "Low Blood Pressure", "Dehydration", "Anxiety", "Inner Ear Disorder"],
}

def call_hf_predict_fallback(user_input: str) -> dict[str, Any]:
    """
    Fallback mock prediction when HF Space is unavailable.
    
    Returns realistic disease predictions based on keyword matching.
    """
    input_lower = user_input.lower()
    
    _LOGGER.info(f"[Mock ML] Using fallback prediction for input: {user_input[:50]}")
    
    # Find matching diseases
    matching_diseases = []
    for symptom, diseases in DISEASE_DATABASE.items():
        if symptom in input_lower:
            matching_diseases.extend(diseases)
    
    # Remove duplicates and limit to top 5
    matching_diseases = list(set(matching_diseases))[:5]
    
    # If no match, return random diseases
    if not matching_diseases:
        all_diseases = []
        for diseases in DISEASE_DATABASE.values():
            all_diseases.extend(diseases)
        matching_diseases = random.sample(list(set(all_diseases)), 3)
    
    # Build response in HF Space format
    response = {
        "predictions": [
            {
                "disease": disease,
                "confidence": round(random.uniform(0.5, 0.95), 2),
                "source": "mock_fallback"
            }
            for disease in matching_diseases
        ],
        "input_text": user_input,
        "model": "fallback_mock"
    }
    
    _LOGGER.info(f"[Mock ML] Returning {len(response['predictions'])} predictions")
    return response


# Sample X-ray analysis responses
XRAY_RESPONSES = {
    "normal": {
        "finding": "Normal chest X-ray",
        "confidence": 0.98,
        "description": "No significant abnormalities detected on frontal and lateral views.",
        "recommendation": "No immediate action required. Continue routine health monitoring."
    },
    "pneumonia": {
        "finding": "Possible pneumonia",
        "confidence": 0.87,
        "description": "Infiltration pattern consistent with pneumonia. Recommend clinical correlation.",
        "recommendation": "Recommend immediate medical evaluation and possible antibiotic therapy."
    },
    "tb": {
        "finding": "Possible tuberculosis",
        "confidence": 0.82,
        "description": "Upper lobe infiltration pattern suggestive of TB. Recommend TB testing.",
        "recommendation": "Urgent referral for TB evaluation and testing."
    },
    "covid": {
        "finding": "Possible COVID-19 pneumonia",
        "confidence": 0.79,
        "description": "Ground-glass opacities and bilateral infiltration pattern.",
        "recommendation": "Recommend RT-PCR testing and clinical evaluation."
    }
}

def analyze_xray_fallback(image_path: str, text: str | None = None) -> dict[str, Any]:
    """
    Fallback mock X-ray analysis when HF Space is unavailable.
    """
    _LOGGER.info(f"[Mock XRay] Using fallback analysis for: {image_path}")
    
    # Random selection for demo
    analysis_type = random.choice(list(XRAY_RESPONSES.keys()))
    response = XRAY_RESPONSES[analysis_type].copy()
    response["image_path"] = image_path
    response["source"] = "mock_fallback"
    
    _LOGGER.info(f"[Mock XRay] Analysis: {response['finding']}")
    return response
