from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.schemas.user import PatientProfile
from app.services.user_service import UserService


@pytest.mark.asyncio
async def test_list_patients_returns_registered_patient_directory() -> None:
    db = AsyncMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.__aiter__.return_value = iter(
        [
            {
                "_id": ObjectId(),
                "email": "patient@example.com",
                "full_name": "Registered Patient",
                "role": "patient",
                "is_active": True,
                "is_blocked": False,
                "created_at": None,
                "updated_at": None,
            }
        ]
    )
    db.users.find = MagicMock(return_value=cursor)

    service = UserService(db)
    service.get_patient_profile = AsyncMock(
        return_value=PatientProfile(
            date_of_birth=None,
            gender="female",
            phone="1234567890",
            emergency_contact="9988776655",
        )
    )

    patients = await service.list_patients()

    db.users.find.assert_called_once_with({"role": "patient", "is_active": True, "is_blocked": False})
    assert len(patients) == 1
    assert patients[0].id
    assert patients[0].name == "Registered Patient"
    assert patients[0].email == "patient@example.com"
    assert patients[0].status == "active"
    assert patients[0].gender == "female"
    assert patients[0].phone == "1234567890"
    assert patients[0].emergency_contact == "9988776655"
