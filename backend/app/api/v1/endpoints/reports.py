from pathlib import Path
from uuid import uuid4

from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.core.deps import get_current_user
from app.db.mongo import get_database
from app.schemas.report import ReportStatusUpdate, TestReportCreate, TestReportPublic
from app.schemas.user import UserPublic
from app.services.report_service import ReportService
from app.services.security_service import SecurityService
from app.services.user_service import UserService

router = APIRouter()

REPORT_STORAGE_DIR = Path(__file__).resolve().parents[4] / "storage" / "reports"
ALLOWED_REPORT_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/jpg", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def _ensure_storage_dir() -> None:
    REPORT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=TestReportPublic)
async def create_report(payload: TestReportCreate, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    user_service = UserService(db)
    patient = await user_service.get_user_by_id(payload.patient_id)
    if not patient or patient.get("role") != "patient":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid patient")
    if current_user.role.value == "patient" and payload.patient_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patients can only upload their own reports")
    if payload.doctor_id:
        doctor = await user_service.get_user_by_id(payload.doctor_id)
        if not doctor or doctor.get("role") != "doctor":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid doctor")
    report = await ReportService(db).create(payload)
    await SecurityService(db).log_audit("report.create", actor_id=current_user.id, target_id=report.id)
    return report


@router.post("/upload", response_model=TestReportPublic)
async def upload_report(
    report_type: str = Form(...),
    patient_id: str | None = Form(default=None),
    doctor_id: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    file: UploadFile = File(...),
    current_user: UserPublic = Depends(get_current_user),
):
    if current_user.role.value == "patient":
        patient_id = current_user.id
    if not patient_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")
    user_service = UserService(get_database())
    patient = await user_service.get_user_by_id(patient_id)
    if not patient or patient.get("role") != "patient":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid patient")
    if doctor_id:
        doctor = await user_service.get_user_by_id(doctor_id)
        if not doctor or doctor.get("role") != "doctor":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid doctor")
    if file.content_type not in ALLOWED_REPORT_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported report file type")

    _ensure_storage_dir()
    report_id = str(uuid4())
    safe_name = file.filename or "report"
    storage_key = f"{report_id}_{safe_name}"
    file_path = REPORT_STORAGE_DIR / storage_key
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file upload")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File too large")
    file_path.write_bytes(content)

    payload = TestReportCreate(
        patient_id=patient_id,
        doctor_id=doctor_id,
        report_type=report_type,
        file_name=safe_name,
        file_path=str(file_path),
        storage_key=storage_key,
        metadata={} if not metadata else {"raw": metadata},
    )
    db = get_database()
    report = await ReportService(db).create(payload)
    await SecurityService(db).log_audit("report.upload", actor_id=current_user.id, target_id=report.id)
    return report


@router.get("", response_model=list[TestReportPublic])
async def list_reports(patient_id: str | None = None, report_type: str | None = None, current_user: UserPublic = Depends(get_current_user)):
    return await ReportService(get_database()).list_for_role(current_user.id, current_user.role.value, patient_id=patient_id, report_type=report_type)


@router.patch("/{report_id}", response_model=TestReportPublic)
async def update_report_status(report_id: str, payload: ReportStatusUpdate, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    report = await ReportService(db).update_status(report_id, payload, current_user.id, current_user.role.value)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    await SecurityService(db).log_audit("report.status_update", actor_id=current_user.id, target_id=report_id)
    return report


@router.get("/{report_id}/download")
async def download_report(report_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    report = await ReportService(get_database()).get_by_id(report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    if current_user.role.value == "patient" and report.patient_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access this report")
    if current_user.role.value == "doctor" and report.doctor_id not in {None, current_user.id}:
        relation = await db.appointments.find_one({"doctor_id": ObjectId(current_user.id), "patient_id": ObjectId(report.patient_id)})
        if not relation:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access this report")
    if current_user.role.value not in {"patient", "doctor", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access this report")
    file_path = Path(report.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(path=str(file_path), filename=report.file_name)


@router.delete("/{report_id}")
async def delete_report(report_id: str, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    service = ReportService(db)
    report = await service.get_by_id(report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    if current_user.role.value == "patient" and report.patient_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete this report")
    if current_user.role.value not in {"patient", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete this report")

    file_path = Path(report.file_path)
    if file_path.exists():
        file_path.unlink()

    deleted = await service.delete_report(report_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    await SecurityService(db).log_audit("report.delete", actor_id=current_user.id, target_id=report_id)
    return {"message": "Report deleted"}
