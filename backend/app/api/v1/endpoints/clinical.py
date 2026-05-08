from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import get_current_user, require_roles
from app.db.mongo import get_database
from app.schemas.clinical import ClinicalNoteCreate, MedicalHistoryItem, MedicationCreate, MedicationPublic, MedicationStatus, MedicationUpdate, PrescriptionCreate, RiskAssessmentPublic, TimelineRecord
from app.schemas.user import Role, UserPublic
from app.services.clinical_service import ClinicalService
from app.services.security_service import SecurityService

router = APIRouter()


async def _doctor_can_access_patient(db, doctor_id: str, patient_id: str) -> bool:
    doctor_oid = ObjectId(doctor_id)
    patient_oid = ObjectId(patient_id)
    appointment = await db.appointments.find_one({"doctor_id": doctor_oid, "patient_id": patient_oid})
    if appointment is not None:
        return True
    # Prescription workflow expects doctors to work with registered patients
    # even before an appointment exists.
    patient_user = await db.users.find_one({"_id": patient_oid, "role": "patient"})
    return patient_user is not None


async def _ensure_patient_access(db, current_user: UserPublic, patient_id: str) -> None:
    if current_user.role == Role.ADMIN:
        return
    if current_user.role == Role.PATIENT and current_user.id == patient_id:
        return
    if current_user.role == Role.DOCTOR and await _doctor_can_access_patient(db, current_user.id, patient_id):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")


@router.put("/family-history")
@router.put("/history")
async def upsert_history(payload: MedicalHistoryItem, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    if current_user.role == Role.DOCTOR and current_user.id != payload.patient_id:
        await _ensure_patient_access(db, current_user, payload.patient_id)
    await ClinicalService(db).upsert_medical_history(payload)
    await SecurityService(db).log_audit("clinical.history_upsert", actor_id=current_user.id, target_id=payload.patient_id)
    return {"message": "Family history updated"}


@router.post("/notes")
async def create_note(payload: ClinicalNoteCreate, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    if current_user.role == Role.DOCTOR and current_user.id != payload.doctor_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor can only create notes for self")
    await _ensure_patient_access(db, current_user, payload.patient_id)
    note_id = await ClinicalService(db).add_note(payload)
    await SecurityService(db).log_audit("clinical.note_create", actor_id=current_user.id, target_id=note_id)
    return {"id": note_id}


@router.delete("/notes/{note_id}")
async def delete_note(note_id: str, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    existing = await db.clinical_notes.find_one({"_id": ObjectId(note_id)})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    if current_user.role == Role.DOCTOR and str(existing["doctor_id"]) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete this note")
    deleted = await ClinicalService(db).delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    await SecurityService(db).log_audit("clinical.note_delete", actor_id=current_user.id, target_id=note_id)
    return {"message": "Note deleted"}


@router.post("/prescriptions")
async def create_prescription(payload: PrescriptionCreate, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    if current_user.role == Role.DOCTOR and current_user.id != payload.doctor_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor can only create prescriptions for self")
    await _ensure_patient_access(db, current_user, payload.patient_id)
    rx_id = await ClinicalService(db).add_prescription(payload)
    await SecurityService(db).log_audit("clinical.prescription_create", actor_id=current_user.id, target_id=rx_id)
    return {"id": rx_id}


@router.post("/medications", response_model=MedicationPublic)
async def create_medication(payload: MedicationCreate, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    if current_user.role == Role.DOCTOR and current_user.id != payload.doctor_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Doctor can only create medications for self")
    await _ensure_patient_access(db, current_user, payload.patient_id)
    medication = await ClinicalService(db).add_medication(payload)
    await SecurityService(db).log_audit("clinical.medication_create", actor_id=current_user.id, target_id=medication.id)
    return medication


@router.get("/medications/current", response_model=list[MedicationPublic])
async def current_medications(patient_id: str | None = None, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    if current_user.role == Role.PATIENT:
        patient_id = current_user.id
    elif not patient_id and current_user.role != Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")
    if not patient_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")
    await _ensure_patient_access(db, current_user, patient_id)
    return await ClinicalService(db).list_medications(patient_id, MedicationStatus.CURRENT)


@router.get("/medications/past", response_model=list[MedicationPublic])
async def past_medications(patient_id: str | None = None, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    if current_user.role == Role.PATIENT:
        patient_id = current_user.id
    elif not patient_id and current_user.role != Role.ADMIN:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")
    if not patient_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")
    await _ensure_patient_access(db, current_user, patient_id)
    return await ClinicalService(db).list_medications(patient_id, MedicationStatus.PAST)


@router.patch("/medications/{medication_id}", response_model=MedicationPublic)
async def update_medication(medication_id: str, payload: MedicationUpdate, current_user: UserPublic = Depends(require_roles(Role.DOCTOR, Role.ADMIN))):
    db = get_database()
    existing = await db.medications.find_one({"_id": ObjectId(medication_id)})
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Medication not found")
    if current_user.role == Role.DOCTOR and str(existing["doctor_id"]) != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify this medication")
    medication = await ClinicalService(db).update_medication_status(medication_id, payload)
    if not medication:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Medication not found")
    await SecurityService(db).log_audit("clinical.medication_update", actor_id=current_user.id, target_id=medication_id)
    return medication


@router.get("/timeline/{patient_id}", response_model=list[TimelineRecord])
async def get_timeline(patient_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    await _ensure_patient_access(db, current_user, patient_id)
    return await ClinicalService(get_database()).timeline(patient_id)


@router.get("/records/me")
async def my_records(current_user: UserPublic = Depends(require_roles(Role.PATIENT))):
    return await ClinicalService(get_database()).unified_records(current_user.id)


@router.get("/records/{patient_id}")
async def patient_records(patient_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    await _ensure_patient_access(db, current_user, patient_id)
    return await ClinicalService(db).unified_records(patient_id)


@router.get("/risk/{patient_id}", response_model=RiskAssessmentPublic)
async def risk_scores(patient_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    await _ensure_patient_access(db, current_user, patient_id)
    assessment = await ClinicalService(db).compute_risk_score(patient_id)
    return assessment
