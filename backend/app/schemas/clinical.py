from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MedicationStatus(str, Enum):
    CURRENT = "current"
    PAST = "past"
    INACTIVE = "inactive"


class MedicalHistoryItem(BaseModel):
    patient_id: str
    diagnosis: str = Field(min_length=2, max_length=400)
    chronic_conditions: list[str] = []
    allergies: list[str] = []


class ClinicalNoteCreate(BaseModel):
    patient_id: str
    doctor_id: str
    note: str = Field(min_length=2, max_length=5000)


class PrescriptionCreate(BaseModel):
    patient_id: str
    doctor_id: str
    medication: str = Field(min_length=2, max_length=250)
    dosage: str = Field(min_length=1, max_length=250)
    instructions: str = Field(min_length=2, max_length=1000)


class TimelineRecord(BaseModel):
    id: str
    record_type: str
    patient_id: str
    doctor_id: str | None = None
    summary: str
    created_at: datetime


class MedicationCreate(BaseModel):
    patient_id: str
    doctor_id: str
    medication_name: str = Field(min_length=2, max_length=250)
    dosage: str = Field(min_length=1, max_length=250)
    instructions: str = Field(min_length=2, max_length=1000)
    status: MedicationStatus = MedicationStatus.CURRENT
    start_date: datetime | None = None
    end_date: datetime | None = None


class MedicationUpdate(BaseModel):
    status: MedicationStatus | None = None
    dosage: str | None = Field(default=None, min_length=1, max_length=250)
    instructions: str | None = Field(default=None, min_length=2, max_length=1000)
    end_date: datetime | None = None


class MedicationPublic(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    medication_name: str
    dosage: str
    instructions: str
    status: MedicationStatus
    start_date: datetime | None = None
    end_date: datetime | None = None
    created_at: datetime
    updated_at: datetime


class RiskAssessmentCreate(BaseModel):
    patient_id: str
    score: float
    summary: str = Field(min_length=2, max_length=500)
    metadata: dict[str, str] = Field(default_factory=dict)


class RiskAssessmentPublic(BaseModel):
    id: str
    patient_id: str
    score: float
    summary: str
    metadata: dict[str, str]
    created_at: datetime
