from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackTarget(str, Enum):
    DOCTOR = "doctor"
    PATIENT = "patient"
    APPOINTMENT = "appointment"
    REPORT = "report"
    APP = "app"


class FeedbackCreate(BaseModel):
    user_id: str
    role: str
    target_type: FeedbackTarget
    target_id: str | None = None
    score: int = Field(ge=1, le=5)
    comment: str = Field(min_length=2, max_length=2000)


class FeedbackCreateInput(BaseModel):
    target_type: FeedbackTarget
    target_id: str | None = None
    score: int = Field(ge=1, le=5)
    comment: str = Field(min_length=2, max_length=2000)


class FeedbackUpdate(BaseModel):
    score: int = Field(ge=1, le=5)
    comment: str = Field(min_length=2, max_length=2000)


class FeedbackPublic(BaseModel):
    id: str
    user_id: str
    user_name: str
    role: str
    target_type: FeedbackTarget
    target_id: str | None = None
    score: int
    comment: str
    created_at: datetime