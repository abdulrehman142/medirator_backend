from datetime import datetime

from pydantic import BaseModel


class SecurityEvent(BaseModel):
    id: str
    event_type: str
    user_id: str | None = None
    detail: str
    created_at: datetime


class AuditLog(BaseModel):
    id: str
    actor_id: str | None = None
    action: str
    target_id: str | None = None
    metadata: dict[str, str] = {}
    created_at: datetime
