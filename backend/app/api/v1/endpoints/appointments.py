from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_current_user
from app.db.mongo import get_database
from app.schemas.appointment import AppointmentCreate, AppointmentPublic, AppointmentStatus, AppointmentUpdate
from app.schemas.user import UserPublic
from app.services.appointment_service import AppointmentService
from app.services.security_service import SecurityService

router = APIRouter()


@router.post("", response_model=AppointmentPublic)
async def create_appointment(payload: AppointmentCreate, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    if current_user.role.value == "patient":
        payload = payload.model_copy(update={"patient_id": current_user.id})
    appointment = await AppointmentService(db).create(payload, current_user.id, current_user.role.value)
    await SecurityService(db).log_audit("appointment.create", actor_id=current_user.id, target_id=appointment.id)
    return appointment


@router.patch("/{appointment_id}", response_model=AppointmentPublic)
async def update_appointment(appointment_id: str, payload: AppointmentUpdate, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    appointment = await AppointmentService(db).update(appointment_id, payload, current_user.id, current_user.role.value)
    if not appointment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")
    action = "appointment.update"
    if payload.status == AppointmentStatus.CANCELED:
        action = "appointment.cancel"
    elif payload.status == AppointmentStatus.COMPLETED:
        action = "appointment.complete"
    elif payload.status == AppointmentStatus.RESCHEDULED:
        action = "appointment.reschedule"
    await SecurityService(db).log_audit(action, actor_id=current_user.id, target_id=appointment_id)
    return appointment


@router.get("", response_model=list[AppointmentPublic])
async def list_appointments(
    start: datetime | None = None,
    end: datetime | None = None,
    current_user: UserPublic = Depends(get_current_user),
):
    return await AppointmentService(get_database()).list(current_user.id, current_user.role.value, start, end)
