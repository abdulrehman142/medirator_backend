from bson import ObjectId
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.feedback import FeedbackCreate, FeedbackPublic, FeedbackTarget, FeedbackUpdate
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow


class FeedbackService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def create(self, payload: FeedbackCreate) -> FeedbackPublic:
        now = utcnow()
        user_doc = await self.db.users.find_one({"_id": ObjectId(payload.user_id)})
        user_name = str((user_doc or {}).get("full_name") or "User")
        res = await self.db.feedback.insert_one(
            {
                "user_id": ObjectId(payload.user_id),
                "user_name": user_name,
                "role": payload.role,
                "target_type": payload.target_type.value,
                "target_id": payload.target_id,
                "score": payload.score,
                "comment": payload.comment,
                "created_at": now,
                "updated_at": now,
            }
        )
        return await self.get_by_id(str(res.inserted_id))

    async def get_by_id(self, feedback_id: str) -> FeedbackPublic | None:
        doc = await self.db.feedback.find_one({"_id": ObjectId(feedback_id)})
        if not doc:
            return None
        d = to_str_id(doc)
        return FeedbackPublic(
            id=d["id"],
            user_id=str(d["user_id"]),
            user_name=str(d.get("user_name") or "User"),
            role=d["role"],
            target_type=FeedbackTarget(d["target_type"]),
            target_id=d.get("target_id"),
            score=d["score"],
            comment=d["comment"],
            created_at=d["created_at"],
        )

    async def list(self, role: str | None = None, target_type: FeedbackTarget | None = None, limit: int = 50) -> list[FeedbackPublic]:
        query: dict = {}
        if role:
            query["role"] = role
        if target_type:
            query["target_type"] = target_type.value
        cursor = self.db.feedback.find(query).sort("created_at", -1).limit(limit)
        items: list[FeedbackPublic] = []
        async for doc in cursor:
            d = to_str_id(doc)
            items.append(
                FeedbackPublic(
                    id=d["id"],
                    user_id=str(d["user_id"]),
                    user_name=str(d.get("user_name") or "User"),
                    role=d["role"],
                    target_type=FeedbackTarget(d["target_type"]),
                    target_id=d.get("target_id"),
                    score=d["score"],
                    comment=d["comment"],
                    created_at=d["created_at"],
                )
            )
        return items

    async def update_own(self, feedback_id: str, user_id: str, payload: FeedbackUpdate) -> FeedbackPublic:
        now = utcnow()
        doc = await self.db.feedback.find_one({"_id": ObjectId(feedback_id)})
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
        if str(doc.get("user_id")) != user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to edit this feedback")

        await self.db.feedback.update_one(
            {"_id": ObjectId(feedback_id)},
            {
                "$set": {
                    "score": payload.score,
                    "comment": payload.comment,
                    "updated_at": now,
                }
            },
        )
        updated = await self.get_by_id(feedback_id)
        if not updated:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
        return updated
