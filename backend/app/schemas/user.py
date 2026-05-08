from datetime import date
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


class Role(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=2, max_length=120)
    role: Role


class UserInDB(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: Role
    hashed_password: str
    is_active: bool = True
    is_blocked: bool = False


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: Role
    is_active: bool


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=120)


class PatientProfile(BaseModel):
    user_id: str | None = None
    full_name: str | None = Field(default=None, min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=0, le=150)
    date_of_birth: date | None = None
    gender: str | None = None
    phone: str | None = None
    blood_group: str | None = None
    allergies: list[str] | None = None
    chronic_diseases: list[str] | None = None
    emergency_contact: str | None = None
    family_history: str | None = None
    family_tree: list[dict] | None = None


class DoctorPatientProfile(BaseModel):
    """Extended patient profile returned to doctors with all available fields."""
    id: str
    name: str
    age: int | None = None
    gender: str | None = None
    contact: str | None = None
    blood_group: str | None = None
    allergies: str | None = None
    chronic_diseases: str | None = None
    emergency_contact: str | None = None
    family_history: str | None = None
    family_tree: list[dict] = Field(default_factory=list)
    doctor_notes: list[dict] = Field(default_factory=list)
    uploaded_documents: list[dict] = Field(default_factory=list)


class DoctorProfile(BaseModel):
    user_id: str | None = None
    full_name: str | None = Field(default=None, min_length=2, max_length=120)
    age: int | None = Field(default=None, ge=0, le=120)
    specialization: str | None = None
    license_number: str | None = None
    years_of_experience: int | None = None
    phone: str | None = None
    is_verified: bool = False


class DoctorDirectoryItem(BaseModel):
    id: str
    display_id: str | None = None
    name: str
    specialization: str | None = None
    status: str
    is_verified: bool = False


class PatientDirectoryItem(BaseModel):
    id: str
    display_id: str | None = None
    name: str
    email: EmailStr
    status: str = "active"
    date_of_birth: date | None = None
    gender: str | None = None
    phone: str | None = None
    emergency_contact: str | None = None
    blood_group: str | None = None
    allergies: list[str] | None = None
    chronic_diseases: list[str] | None = None
    family_history: str | None = None
