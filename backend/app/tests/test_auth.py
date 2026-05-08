from app.core.security import create_access_token, decode_token, hash_password, verify_password


def test_password_hash_and_verify() -> None:
    plain = "SecurePass123!"
    hashed = hash_password(plain)
    assert hashed != plain
    assert verify_password(plain, hashed)


def test_access_token_encode_decode() -> None:
    token = create_access_token(subject="user-id", role="patient")
    payload = decode_token(token)
    assert payload["sub"] == "user-id"
    assert payload["role"] == "patient"
    assert payload["type"] == "access"
