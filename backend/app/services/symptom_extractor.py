"""
Symptom extraction logic for AI Risk Analysis.
Converts free-text symptom descriptions into structured feature vectors.
"""


def extract_symptoms(
    text: str,
    symptoms_list: list[str],
) -> list[int]:
    """
    Converts a free-text symptom description into a binary feature vector.

    Args:
        text: Free-text symptom description
        symptoms_list: Ordered vocabulary list of symptoms

    Returns:
        Binary vector of length len(symptoms_list) where position i is 1
        if symptoms_list[i] appears (case-insensitive) in the text, 0 otherwise.
    """
    # Normalize input text to lowercase
    normalized_text = text.lower()

    # Build feature vector
    feature_vector = []
    for symptom in symptoms_list:
        # Check if symptom appears as a substring (case-insensitive)
        if symptom.lower() in normalized_text:
            feature_vector.append(1)
        else:
            feature_vector.append(0)

    return feature_vector
