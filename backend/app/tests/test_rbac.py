import pytest
from fastapi import HTTPException

from app.core.deps import require_roles
from app.schemas.user import Role, UserPublic


@pytest.mark.asyncio
async def test_require_roles_allows_valid_role() -> None:
    guard = require_roles(Role.ADMIN)
    user = UserPublic(id="1", email="admin@example.com", full_name="Admin", role=Role.ADMIN, is_active=True)
    result = await guard(user)
    assert result.role == Role.ADMIN


@pytest.mark.asyncio
async def test_require_roles_blocks_invalid_role() -> None:
    guard = require_roles(Role.DOCTOR)
    user = UserPublic(id="2", email="patient@example.com", full_name="Patient", role=Role.PATIENT, is_active=True)

    with pytest.raises(HTTPException) as exc:
        await guard(user)

    assert exc.value.status_code == 403
