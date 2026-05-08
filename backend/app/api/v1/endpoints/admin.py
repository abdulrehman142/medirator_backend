from fastapi import APIRouter, Depends, Response, status

from app.core.deps import require_roles
from app.db.redis import get_redis
from app.db.mongo import get_database
from app.schemas.admin import (
    DashboardMetrics,
    DoctorAdminItem,
    DoctorAdminListResponse,
    DoctorAdminPatchRequest,
    PatientAdminItem,
    PatientAdminListResponse,
    PatientAdminPatchRequest,
    UserBlockRequest,
    CompletedAppointmentsResponse,
)
from app.schemas.user import Role, UserPublic
from app.services.admin_service import AdminService
from app.services.security_service import SecurityService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/metrics", response_model=DashboardMetrics)
async def get_metrics(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    return await AdminService(get_database()).dashboard_metrics()


@router.get("/insights")
async def get_insights(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    return await AdminService(get_database()).basic_insights()


@router.get("/analytics")
async def get_analytics(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    return await AdminService(get_database()).analytics_snapshot()


@router.get("/completed-appointments", response_model=CompletedAppointmentsResponse)
async def get_completed_appointments(limit: int = 10, current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    appts = await AdminService(get_database()).recent_completed_appointments(limit=limit)
    return CompletedAppointmentsResponse(appointments=appts)


@router.get("/doctors")
async def get_doctors(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    doctors = await AdminService(get_database()).list_doctors()
    return DoctorAdminListResponse(doctors=doctors)


@router.patch("/doctors/{doctor_id}", response_model=DoctorAdminItem)
async def patch_doctor(
    doctor_id: str,
    payload: DoctorAdminPatchRequest,
    current_user: UserPublic = Depends(require_roles(Role.ADMIN)),
):
    return await AdminService(get_database()).patch_doctor(doctor_id, payload)


@router.get("/patients")
async def get_patients(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    patients = await AdminService(get_database()).list_patients()
    return PatientAdminListResponse(patients=patients)


@router.patch("/patients/{patient_id}", response_model=PatientAdminItem)
async def patch_patient(
    patient_id: str,
    payload: PatientAdminPatchRequest,
    current_user: UserPublic = Depends(require_roles(Role.ADMIN)),
):
    return await AdminService(get_database()).patch_patient(patient_id, payload)


@router.delete("/patients/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(patient_id: str, current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    await AdminService(get_database()).delete_patient(patient_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/reports")
async def get_reports(current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    return await AdminService(get_database()).list_reports() or []


@router.post("/users/block")
async def block_user(payload: UserBlockRequest, current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    db = get_database()
    await UserService(db).set_user_blocked(payload.user_id, payload.blocked)
    await SecurityService(db).log_audit("admin.user_block_toggle", actor_id=current_user.id, target_id=payload.user_id)
    return {"message": "User block status updated"}


@router.post("/users/reset-lock/{email}")
async def reset_user_lock(email: str, current_user: UserPublic = Depends(require_roles(Role.ADMIN))):
    redis_client = get_redis()
    await redis_client.delete(f"failed-login:{email.lower()}")
    db = get_database()
    await db.users.update_one({"email": email.lower()}, {"$set": {"is_blocked": False}})
    await SecurityService(db).log_audit("admin.user_lock_reset", actor_id=current_user.id, target_id=email)
    return {"message": "User lock reset"}
