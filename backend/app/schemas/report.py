from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ReportStatus(str, Enum):
    UPLOADED = "uploaded"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class TestReportCreate(BaseModel):
    patient_id: str
    doctor_id: str | None = None
    report_type: str = Field(min_length=2, max_length=120)
    file_name: str = Field(min_length=1, max_length=255)
    file_path: str = Field(min_length=1, max_length=500)
    storage_key: str | None = None
    status: ReportStatus = ReportStatus.UPLOADED
    metadata: dict[str, str] = Field(default_factory=dict)


class ReportUploadResponse(BaseModel):
    id: str
    patient_id: str
    doctor_id: str | None = None
    report_type: str
    file_name: str
    file_path: str
    storage_key: str | None = None
    status: ReportStatus
    metadata: dict[str, str]
    created_at: datetime


class ReportStatusUpdate(BaseModel):
    status: ReportStatus | None = None
    metadata: dict[str, str] | None = None


class TestReportPublic(BaseModel):
    id: str
    patient_id: str
    doctor_id: str | None = None
    report_type: str
    file_name: str
    file_path: str
    storage_key: str | None = None
    status: ReportStatus = ReportStatus.UPLOADED
    metadata: dict[str, str]
    created_at: datetime
