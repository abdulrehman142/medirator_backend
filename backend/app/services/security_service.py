from motor.motor_asyncio import AsyncIOMotorDatabase

from app.utils.time import utcnow


class SecurityService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def log_security_event(self, event_type: str, detail: str, user_id: str | None = None) -> None:
        await self.db.security_events.insert_one(
            {
                "event_type": event_type,
                "detail": detail,
                "user_id": user_id,
                "created_at": utcnow(),
            }
        )

    async def log_audit(self, action: str, actor_id: str | None = None, target_id: str | None = None, metadata: dict | None = None) -> None:
        await self.db.audit_logs.insert_one(
            {
                "action": action,
                "actor_id": actor_id,
                "target_id": target_id,
                "metadata": metadata or {},
                "created_at": utcnow(),
            }
        )
