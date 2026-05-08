from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.db.mongo import get_database
from app.schemas.user import Role, UserPublic
from app.services.user_service import UserService
from app.core.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPublic:
    try:
        payload = decode_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token subject")

    db = get_database()
    user_service = UserService(db)
    user = await user_service.get_user_public_by_id(user_id)
    user_doc = await user_service.get_user_by_id(user_id)
    if not user or not user.is_active or bool((user_doc or {}).get("is_blocked", False)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or missing user")

    return user


def require_roles(*roles: Role) -> Callable[[UserPublic], UserPublic]:
    async def role_guard(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return role_guard
