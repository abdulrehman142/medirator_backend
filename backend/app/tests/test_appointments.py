from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi import HTTPException

from app.schemas.appointment import AppointmentCreate, AppointmentStatus
from app.services.appointment_service import AppointmentService


@pytest.mark.asyncio
async def test_create_appointment_calls_insert() -> None:
    db = AsyncMock()
    inserted_id = ObjectId()
    db.appointments.insert_one = AsyncMock()
    db.appointments.insert_one.return_value.inserted_id = inserted_id
    db.appointments.find_one = AsyncMock(
        return_value={
            "_id": inserted_id,
            "patient_id": ObjectId(),
            "doctor_id": ObjectId(),
            "reason": "Follow-up",
            "scheduled_for": datetime.now(UTC),
            "status": AppointmentStatus.SCHEDULED.value,
            "notes": None,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
    )

    service = AppointmentService(db)
    service.user_service.get_user_by_id = AsyncMock(
        side_effect=[
            {"id": str(ObjectId()), "role": "patient", "is_active": True, "is_blocked": False},
            {"id": str(ObjectId()), "role": "doctor", "is_active": True, "is_blocked": False},
        ]
    )
    payload = AppointmentCreate(
        patient_id=str(ObjectId()),
        doctor_id=str(ObjectId()),
        reason="Follow-up",
        scheduled_for=datetime.now(UTC),
    )

    appointment = await service.create(payload)

    assert appointment.id == str(inserted_id)
    assert appointment.status == AppointmentStatus.SCHEDULED
    db.appointments.insert_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_appointment_rejects_invalid_doctor_or_patient() -> None:
    db = AsyncMock()
    service = AppointmentService(db)
    service.user_service.get_user_by_id = AsyncMock(return_value=None)

    payload = AppointmentCreate(
        patient_id=str(ObjectId()),
        doctor_id=str(ObjectId()),
        reason="Consultation",
        scheduled_for=datetime.now(UTC),
    )

    with pytest.raises(HTTPException) as exc:
        await service.create(payload)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid patient"


def test_appointment_create_accepts_camelcase_and_datetime_local_input() -> None:
    payload = AppointmentCreate.model_validate(
        {
            "patientId": str(ObjectId()),
            "doctorId": str(ObjectId()),
            "reason": "Consultation",
            "scheduledFor": "2026-04-27T10:30",
        }
    )

    assert payload.patient_id is not None
    assert payload.doctor_id is not None
    assert payload.scheduled_for.strftime("%Y-%m-%dT%H:%M") == "2026-04-27T10:30"
