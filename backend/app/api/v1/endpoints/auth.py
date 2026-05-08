from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.db.mongo import get_database
from app.schemas.auth import ForgotPasswordRequest, LoginRequest, RefreshRequest, RegisterRequest, ResetPasswordRequest, TokenPair
from app.services.auth_service import AuthService

router = APIRouter()


@router.post("/register", response_model=TokenPair)
async def register(payload: RegisterRequest):
    service = AuthService(get_database())
    return await service.register(payload)


@router.post("/login", response_model=TokenPair)
async def login(payload: LoginRequest):
    service = AuthService(get_database())
    return await service.login(payload)


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest):
    service = AuthService(get_database())
    return await service.refresh(payload.refresh_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest):
    service = AuthService(get_database())
    await service.logout(payload.refresh_token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest):
    service = AuthService(get_database())
    token_or_msg = await service.forgot_password(payload.email)
    return {"message": "Password reset initiated", "debug_token": token_or_msg}


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest):
    service = AuthService(get_database())
    try:
        await service.reset_password(payload.token, payload.new_password)
    except HTTPException:
        raise
    return {"message": "Password updated successfully"}
