from fastapi import APIRouter, Header, HTTPException, Response, status
from pymongo.errors import ServerSelectionTimeoutError

from app.db.mongo import get_database
from app.schemas.auth import ForgotPasswordRequest, LoginRequest, RefreshRequest, RegisterRequest, ResetPasswordRequest, TokenPair
from app.services.auth_service import AuthService

router = APIRouter()


def _extract_refresh_token(payload: RefreshRequest | None, authorization: str | None) -> str:
    if payload and payload.refresh_token:
        return payload.refresh_token

    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value.split(" ", 1)[1].strip()
        return value

    raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.post("/register", response_model=TokenPair)
async def register(payload: RegisterRequest):
    try:
        service = AuthService(get_database())
        return await service.register(payload)
    except HTTPException:
        raise
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again later.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(exc)}")


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest):
    try:
        service = AuthService(get_database())
        return await service.login(payload)
    except HTTPException:
        raise
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again later.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(exc)}")


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest | None = None, authorization: str | None = Header(default=None)):
    try:
        service = AuthService(get_database())
        refresh_token = _extract_refresh_token(payload, authorization)
        return await service.refresh(refresh_token)
    except HTTPException:
        raise
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again later.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(exc)}")


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest | None = None, authorization: str | None = Header(default=None)):
    try:
        service = AuthService(get_database())
        refresh_token = _extract_refresh_token(payload, authorization)
        await service.logout(refresh_token)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again later.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(exc)}")


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    try:
        service = AuthService(get_database())
        token_or_msg = await service.forgot_password(payload.email)
        return {"message": "Password reset initiated", "debug_token": token_or_msg}
    except HTTPException:
        raise
    except ServerSelectionTimeoutError:
        raise HTTPException(status_code=503, detail="Database unavailable. Please try again later.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Password reset failed: {str(exc)}")


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    service = AuthService(get_database())
    try:
        await service.reset_password(payload.token, payload.new_password)
    except HTTPException:
        raise
    return {"message": "Password updated successfully"}
