"""Utility functions for formatting IDs consistently across the application."""


def format_patient_id(mongo_id: str) -> str:
    """Format MongoDB patient ID to display format: PAT-XXXX-XXXX."""
    if not mongo_id or len(mongo_id) < 8:
        return mongo_id
    # Take first 8 hex characters, split into 4+4, uppercase, add PAT- prefix
    hex_part = mongo_id[:8].upper()
    return f"PAT-{hex_part[:4]}-{hex_part[4:8]}"


def format_doctor_id(mongo_id: str) -> str:
    """Format MongoDB doctor ID to display format: DR-XXXX-XXXX."""
    if not mongo_id or len(mongo_id) < 8:
        return mongo_id
    # Take first 8 hex characters, split into 4+4, uppercase, add DR- prefix
    hex_part = mongo_id[:8].upper()
    return f"DR-{hex_part[:4]}-{hex_part[4:8]}"


def decode_formatted_id(formatted_id: str, expected_prefix: str | None = None) -> str | None:
    """
    Decode formatted ID back to MongoDB format.
    
    Handles both formatted IDs (e.g., PAT-69EC-A05C) and raw MongoDB IDs.
    If expected_prefix is provided, validates that the formatted ID starts with it.
    Returns the MongoDB ID or None if invalid.
    """
    if not formatted_id:
        return None
    
    # If it's a formatted ID with prefix
    if "-" in formatted_id:
        parts = formatted_id.split("-")
        if len(parts) != 3:
            return None
        
        prefix, part1, part2 = parts
        
        # Validate prefix if provided
        if expected_prefix and prefix != expected_prefix:
            return None
        
        # Reconstruct the original hex ID (first 8 chars)
        reconstructed = (part1 + part2).lower()
        
        # Validate it's valid hex
        try:
            int(reconstructed, 16)
            return reconstructed
        except ValueError:
            return None
    
    # Otherwise assume it's a raw MongoDB ID
    return formatted_id

