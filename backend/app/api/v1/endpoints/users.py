from fastapi import APIRouter, Depends

from app.core.deps import get_current_user, require_roles
from app.db.mongo import get_database
from app.schemas.user import DoctorDirectoryItem, DoctorPatientProfile, DoctorProfile, PatientDirectoryItem, PatientProfile, Role, UserPublic, UserUpdate
from app.services.security_service import SecurityService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/me", response_model=UserPublic)
async def my_profile(current_user: UserPublic = Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserPublic)
async def update_profile(payload: UserUpdate, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    service = UserService(db)
    updated = await service.update_user(current_user.id, payload)
    await SecurityService(db).log_audit("user.profile_update", actor_id=current_user.id)
    return updated


@router.get("/patients/me", response_model=PatientProfile | None)
async def get_my_patient_profile(current_user: UserPublic = Depends(require_roles(Role.PATIENT))):
    service = UserService(get_database())
    return await service.get_patient_profile(current_user.id)


@router.put("/patients/me", response_model=PatientProfile)
async def upsert_my_patient_profile(payload: PatientProfile, current_user: UserPublic = Depends(require_roles(Role.PATIENT))):
    db = get_database()
    service = UserService(db)
    profile = await service.upsert_patient_profile(current_user.id, payload)
    await SecurityService(db).log_audit("patient.profile_upsert", actor_id=current_user.id)
    return profile


@router.get("/patients", response_model=list[PatientDirectoryItem])
async def list_patients(current_user: UserPublic = Depends(require_roles(Role.DOCTOR))):
    patients = await UserService(get_database()).list_patients()
    return [patient.model_dump() for patient in patients] if patients else []


@router.get("/patients/{patient_user_id}/profile", response_model=DoctorPatientProfile | None)
async def get_patient_profile_for_doctor(
    patient_user_id: str,
    current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN)),
):
    _ = current_user
    service = UserService(get_database())
    return await service.get_doctor_patient_profile(patient_user_id)


@router.get("/doctors/me", response_model=DoctorProfile | None)
async def get_my_doctor_profile(current_user: UserPublic = Depends(require_roles(Role.DOCTOR))):
    service = UserService(get_database())
    return await service.get_doctor_profile(current_user.id)


@router.put("/doctors/me", response_model=DoctorProfile)
async def upsert_my_doctor_profile(payload: DoctorProfile, current_user: UserPublic = Depends(require_roles(Role.DOCTOR))):
    db = get_database()
    service = UserService(db)
    profile = await service.upsert_doctor_profile(current_user.id, payload)
    await SecurityService(db).log_audit("doctor.profile_upsert", actor_id=current_user.id)
    return profile


@router.get("/doctors/directory", response_model=list[DoctorDirectoryItem])
async def list_doctor_directory(current_user: UserPublic = Depends(require_roles(Role.PATIENT, Role.DOCTOR))):
    return await UserService(get_database()).list_doctor_directory() or []


@router.get("/doctors", response_model=list[dict])
async def list_doctors(current_user: UserPublic = Depends(require_roles(Role.PATIENT, Role.DOCTOR, Role.ADMIN))):
    doctors = await UserService(get_database()).list_doctor_directory()
    return [doctor.model_dump() for doctor in doctors] if doctors else []
