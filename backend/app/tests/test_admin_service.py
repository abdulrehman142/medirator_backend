from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from app.schemas.admin import DoctorAdminPatchRequest, DoctorManagementRole, DoctorManagementStatus, DoctorVerificationStatus
from app.services.admin_service import AdminService


@pytest.mark.asyncio
async def test_list_doctors_returns_management_shape() -> None:
    db = AsyncMock()
    doctor_id = ObjectId()

    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.__aiter__.return_value = iter(
        [
            {
                "_id": doctor_id,
                "full_name": "Dr. Example",
                "role": "doctor",
                "is_active": False,
                "is_blocked": True,
                "created_at": None,
            }
        ]
    )
    db.users.find = MagicMock(return_value=cursor)
    db.doctors.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(),
            "user_id": doctor_id,
            "specialization": "Cardiology",
            "verification": "approved",
            "admin_role": "Supervisor",
        }
    )

    service = AdminService(db)
    doctors = await service.list_doctors()

    assert len(doctors) == 1
    assert doctors[0].name == "Dr. Example"
    assert doctors[0].specialization == "Cardiology"
    assert doctors[0].status == DoctorManagementStatus.SUSPENDED
    assert doctors[0].verification == DoctorVerificationStatus.APPROVED
    assert doctors[0].role == DoctorManagementRole.SUPERVISOR


@pytest.mark.asyncio
async def test_patch_doctor_approve_sets_active_and_verified() -> None:
    db = AsyncMock()
    doctor_id = ObjectId()

    db.users.find_one = AsyncMock(
        side_effect=[
            {
                "_id": doctor_id,
                "full_name": "Dr. Patch",
                "role": "doctor",
                "is_active": False,
                "is_blocked": True,
            },
            {
                "_id": doctor_id,
                "full_name": "Dr. Patch",
                "role": "doctor",
                "is_active": True,
                "is_blocked": False,
            },
        ]
    )
    db.users.update_one = AsyncMock()
    db.doctors.update_one = AsyncMock()
    db.doctors.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(),
            "user_id": doctor_id,
            "specialization": "Neurology",
            "verification": "approved",
            "admin_role": "Senior Doctor",
        }
    )

    service = AdminService(db)
    updated = await service.patch_doctor(
        str(doctor_id),
        DoctorAdminPatchRequest(verification="Approved"),
    )

    assert updated.status == DoctorManagementStatus.ACTIVE
    assert updated.verification == DoctorVerificationStatus.APPROVED
    db.users.update_one.assert_awaited_once()
    db.doctors.update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_patch_doctor_reject_sets_suspended_and_rejected() -> None:
    db = AsyncMock()
    doctor_id = ObjectId()

    db.users.find_one = AsyncMock(
        side_effect=[
            {
                "_id": doctor_id,
                "full_name": "Dr. Patch",
                "role": "doctor",
                "is_active": True,
                "is_blocked": False,
            },
            {
                "_id": doctor_id,
                "full_name": "Dr. Patch",
                "role": "doctor",
                "is_active": False,
                "is_blocked": True,
            },
        ]
    )
    db.users.update_one = AsyncMock()
    db.doctors.update_one = AsyncMock()
    db.doctors.find_one = AsyncMock(
        return_value={
            "_id": ObjectId(),
            "user_id": doctor_id,
            "specialization": "Neurology",
            "verification": "rejected",
            "admin_role": "Doctor",
        }
    )

    service = AdminService(db)
    updated = await service.patch_doctor(
        str(doctor_id),
        DoctorAdminPatchRequest(verification="Rejected"),
    )

    assert updated.status == DoctorManagementStatus.SUSPENDED
    assert updated.verification == DoctorVerificationStatus.REJECTED
    db.users.update_one.assert_awaited_once()
    db.doctors.update_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_patient_deletes_profile_and_user() -> None:
    db = AsyncMock()
    patient_id = ObjectId()

    db.users.find_one = AsyncMock(
        return_value={
            "_id": patient_id,
            "full_name": "Patient Test",
            "role": "patient",
        }
    )
    db.patients.delete_many = AsyncMock()
    delete_result = MagicMock()
    delete_result.deleted_count = 1
    db.users.delete_one = AsyncMock(return_value=delete_result)

    service = AdminService(db)
    deleted = await service.delete_patient(str(patient_id))

    assert deleted is True
    db.patients.delete_many.assert_awaited_once()
    db.users.delete_one.assert_awaited_once()
