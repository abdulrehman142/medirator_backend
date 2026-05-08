from datetime import datetime
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    RESCHEDULED = "rescheduled"
    CANCELED = "canceled"
    COMPLETED = "completed"


class AppointmentCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    patient_id: str | None = Field(default=None, validation_alias=AliasChoices("patient_id", "patientId"))
    doctor_id: str = Field(validation_alias=AliasChoices("doctor_id", "doctorId"))
    reason: str = Field(min_length=2, max_length=1000)
    scheduled_for: datetime = Field(validation_alias=AliasChoices("scheduled_for", "scheduledFor"))

    @field_validator("scheduled_for", mode="before")
    @classmethod
    def parse_scheduled_for(cls, value):
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            candidate = candidate.replace("Z", "+00:00")
            parse_formats = (
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%b %d, %Y, %I:%M %p",
            )
            for fmt in parse_formats:
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(candidate)
            except ValueError as exc:
                raise ValueError("Invalid appointment time format") from exc
        return value


class AppointmentUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    scheduled_for: datetime | None = Field(default=None, validation_alias=AliasChoices("scheduled_for", "scheduledFor"))
    status: AppointmentStatus | None = None
    notes: str | None = Field(default=None, max_length=1000)
    reason: str | None = Field(default=None, min_length=2, max_length=1000)

    @field_validator("scheduled_for", mode="before")
    @classmethod
    def parse_scheduled_for(cls, value):
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            candidate = candidate.replace("Z", "+00:00")
            parse_formats = (
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%b %d, %Y, %I:%M %p",
            )
            for fmt in parse_formats:
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(candidate)
            except ValueError as exc:
                raise ValueError("Invalid appointment time format") from exc
        return value


class AppointmentPublic(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    reason: str
    scheduled_for: datetime
    scheduled_for_display: str | None = None
    status: AppointmentStatus
    notes: str | None = None
    created_at: datetime
    updated_at: datetime
