from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DashboardMetrics(BaseModel):
    total_users: int
    total_appointments: int
    active_doctors: int


class UserBlockRequest(BaseModel):
    user_id: str
    blocked: bool


class UserSummary(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    is_active: bool


class AdminReportSummary(BaseModel):
    id: str
    patient_id: str
    doctor_id: str | None = None
    report_type: str
    file_name: str
    status: str
    created_at: str


class DoctorManagementRole(str, Enum):
    DOCTOR = "Doctor"
    SENIOR_DOCTOR = "Senior Doctor"
    SUPERVISOR = "Supervisor"


class DoctorManagementStatus(str, Enum):
    ACTIVE = "Active"
    SUSPENDED = "Suspended"


class DoctorVerificationStatus(str, Enum):
    APPROVED = "Approved"
    PENDING = "Pending"
    REJECTED = "Rejected"


class PatientManagementStatus(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"


class DoctorAdminItem(BaseModel):
    id: str
    name: str
    specialization: str | None = None
    status: DoctorManagementStatus
    verification: DoctorVerificationStatus
    role: DoctorManagementRole = DoctorManagementRole.DOCTOR


class PatientAdminItem(BaseModel):
    id: str
    user_id: str | None = None
    name: str
    age: int | None = Field(default=None, ge=0, le=120)
    status: PatientManagementStatus
    flagged_critical: bool = False
    family_history: str | None = None
    gender: str | None = None
    phone: str | None = None
    blood_group: str | None = None
    allergies: str | None = None
    chronic_diseases: str | None = None
    emergency_contact: str | None = None


class CompletedAppointmentSummary(BaseModel):
    id: str
    patient_id: str | None = None
    patient_name: str | None = None
    doctor_id: str | None = None
    doctor_name: str | None = None
    scheduled_for: str | None = None
    updated_at: str | None = None
    status: str | None = None


class CompletedAppointmentsResponse(BaseModel):
    appointments: list[CompletedAppointmentSummary]


class DoctorAdminListResponse(BaseModel):
    doctors: list[DoctorAdminItem]


class PatientAdminListResponse(BaseModel):
    patients: list[PatientAdminItem]


class DoctorAdminPatchRequest(BaseModel):
    status: DoctorManagementStatus | None = None
    verification: DoctorVerificationStatus | None = None
    role: DoctorManagementRole | None = DoctorManagementRole.DOCTOR

    @field_validator("status", "verification", "role", mode="before")
    @classmethod
    def normalize_value(cls, value: str | None):
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "active": "Active",
                "suspended": "Suspended",
                "approved": "Approved",
                "pending": "Pending",
                "rejected": "Rejected",
                "doctor": "Doctor",
                "senior doctor": "Senior Doctor",
                "supervisor": "Supervisor",
            }
            return mapping.get(normalized, value)
        return value


class PatientAdminPatchRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=0, le=120)
    status: PatientManagementStatus | None = None
    flagged_critical: bool | None = None
    family_history: str | None = Field(default=None, max_length=5000)

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str | None):
        if value is None:
            return None
        if isinstance(value, str):
            mapping = {
                "active": "Active",
                "inactive": "Inactive",
            }
            return mapping.get(value.strip().lower(), value)
        return value
