import uuid
from datetime import timedelta

from bson import ObjectId
from fastapi import HTTPException, status
from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.core.security import create_access_token, create_password_reset_token, create_refresh_token, hash_password, verify_password
from app.db.redis import get_redis
from app.schemas.auth import LoginRequest, RegisterRequest, TokenPair
from app.schemas.user import UserCreate
from app.services.security_service import SecurityService
from app.services.user_service import UserService
from app.utils.time import utcnow


class AuthService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.user_service = UserService(db)
        self.security_service = SecurityService(db)

    async def register(self, payload: RegisterRequest) -> TokenPair:
        existing = await self.user_service.get_user_by_email(payload.email)
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

        try:
            user = await self.user_service.create_user(
                UserCreate(
                    email=payload.email,
                    password=payload.password,
                    full_name=payload.full_name,
                    role=payload.role,
                ),
                hash_password(payload.password),
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid registration payload") from exc

        if user.role.value == "doctor":
            # Doctors can only access after explicit admin approval.
            await self.db.users.update_one(
                {"_id": ObjectId(user.id)},
                {"$set": {"is_active": False, "updated_at": utcnow()}},
            )
            await self.db.doctors.update_one(
                {"user_id": ObjectId(user.id)},
                {"$set": {"verification": "Pending", "is_verified": False, "updated_at": utcnow()}},
            )
            await self.security_service.log_audit(action="auth.register_pending_doctor", actor_id=user.id)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Doctor account created. Await admin approval before login.",
            )

        token_pair = await self._issue_tokens(user.id, user.role.value)
        await self.security_service.log_audit(action="auth.register", actor_id=user.id)
        return token_pair

    async def login(self, payload: LoginRequest) -> TokenPair:
        settings = get_settings()
        redis_client = get_redis()
        normalized_email = payload.email.strip().lower()
        fail_key = f"failed-login:{normalized_email}"

        async def _redis_call(method: str, *args, **kwargs) -> None:
            try:
                await getattr(redis_client, method)(*args, **kwargs)
            except Exception:
                # Fail open if Redis is unavailable so auth itself still works.
                return

        user = await self.user_service.get_user_by_email(normalized_email)
        if not user:
            await _redis_call("incr", fail_key)
            await _redis_call("expire", fail_key, settings.block_duration_minutes * 60)
            await self.security_service.log_security_event("failed_login", "Unknown email", None)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        if user.get("is_blocked"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is blocked")

        if user.get("role") == "doctor":
            doctor_profile = await self.db.doctors.find_one({"user_id": ObjectId(user["id"])})
            verification = str((doctor_profile or {}).get("verification", "")).strip().lower()
            is_verified = bool((doctor_profile or {}).get("is_verified", False))
            is_approved = verification == "approved" or is_verified
            if not is_approved:
                if verification == "rejected":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Account rejected by admin. Please contact support.",
                    )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Doctor account is pending admin approval.",
                )

        if not user.get("is_active", True):
            role = user.get("role")
            if role == "doctor":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Doctor account is inactive.",
                )
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

        if not verify_password(payload.password, user["hashed_password"]):
            try:
                failures = await redis_client.incr(fail_key)
                await redis_client.expire(fail_key, settings.block_duration_minutes * 60)
            except Exception:
                failures = 0
            if failures >= settings.failed_login_limit:
                await self.db.users.update_one(
                    {"_id": ObjectId(user["id"])},
                    {"$set": {"is_blocked": True, "updated_at": utcnow()}},
                )
                await self.security_service.log_security_event("account_blocked", "Too many failed logins", user["id"])
            await self.security_service.log_security_event("failed_login", "Wrong password", user["id"])
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        try:
            await redis_client.delete(fail_key)
        except Exception:
            pass
        token_pair = await self._issue_tokens(user["id"], user["role"])
        await self.security_service.log_audit(action="auth.login", actor_id=user["id"])
        return token_pair

    async def refresh(self, refresh_token: str) -> TokenPair:
        settings = get_settings()
        try:
            payload = jwt.decode(refresh_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

        jti = payload.get("jti")
        user_id = payload.get("sub")
        if not jti or not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed token")

        doc = await self.db.refresh_tokens.find_one({"jti": jti, "revoked": False})
        if not doc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked")

        await self.db.refresh_tokens.update_one({"_id": doc["_id"]}, {"$set": {"revoked": True, "updated_at": utcnow()}})

        user = await self.user_service.get_user_public_by_id(user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        return await self._issue_tokens(user_id, user.role.value)

    async def logout(self, refresh_token: str) -> None:
        settings = get_settings()
        payload = jwt.decode(refresh_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        jti = payload.get("jti")
        if jti:
            await self.db.refresh_tokens.update_one({"jti": jti}, {"$set": {"revoked": True, "updated_at": utcnow()}})

    async def forgot_password(self, email: str) -> str:
        user = await self.user_service.get_user_by_email(email)
        if not user:
            return "If the account exists, reset instructions have been generated."

        token = create_password_reset_token(user["id"])
        await self.db.security_events.insert_one(
            {
                "event_type": "password_reset_requested",
                "detail": "Reset token generated",
                "user_id": ObjectId(user["id"]),
                "token": token,
                "created_at": utcnow(),
            }
        )
        return token

    async def reset_password(self, token: str, new_password: str) -> None:
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid reset token") from exc

        if payload.get("type") != "password_reset":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token missing subject")

        await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"hashed_password": hash_password(new_password), "updated_at": utcnow()}},
        )

    async def _issue_tokens(self, user_id: str, role: str) -> TokenPair:
        jti = str(uuid.uuid4())
        access_token = create_access_token(subject=user_id, role=role)
        refresh_token = create_refresh_token(subject=user_id, jti=jti)

        settings = get_settings()
        await self.db.refresh_tokens.insert_one(
            {
                "user_id": ObjectId(user_id),
                "jti": jti,
                "token": refresh_token,
                "revoked": False,
                "expires_at": utcnow() + timedelta(days=settings.refresh_token_expire_days),
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
        )
        return TokenPair(access_token=access_token)
