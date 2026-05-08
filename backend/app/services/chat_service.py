import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.collections import COLLECTIONS
from app.schemas.chat_history import ChatMessageDB, ChatMessageResponse

class ChatService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[COLLECTIONS["chat_history"]]

    async def get_user_history(self, user_id: str, limit: int = 50) -> List[ChatMessageResponse]:
        cursor = self.collection.find({"user_id": user_id}).sort("timestamp", 1).limit(limit)
        messages = []
        async for doc in cursor:
            messages.append(
                ChatMessageResponse(
                    id=doc["_id"],
                    user_message=doc["user_message"],
                    answer=doc.get("ai_response", ""),
                    confidence=doc.get("confidence"),
                    model_used=doc.get("model_used", "Unknown"),
                    timestamp=doc.get("timestamp", datetime.utcnow())
                )
            )
        return messages

    async def save_interaction(
        self,
        user_id: str,
        session_id: str,
        input_type: str,
        user_message: str,
        ai_response: str,
        confidence: Optional[float],
        model_used: str,
        files_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> ChatMessageResponse:
        doc_id = uuid.uuid4().hex
        timestamp = datetime.utcnow()
        
        chat_doc = {
            "_id": doc_id,
            "user_id": user_id,
            "session_id": session_id,
            "input_type": input_type,
            "user_message": user_message,
            "files_metadata": files_metadata,
            "ai_response": ai_response,
            "confidence": confidence,
            "model_used": model_used,
            "timestamp": timestamp
        }
        
        await self.collection.insert_one(chat_doc)
        
        return ChatMessageResponse(
            id=doc_id,
            user_message=user_message,
            answer=ai_response,
            confidence=confidence,
            model_used=model_used,
            timestamp=timestamp
        )

    async def delete_user_history(self, user_id: str) -> Dict[str, Any]:
        """Delete all chat history for a user."""
        result = await self.collection.delete_many({"user_id": user_id})
        return {
            "status": "success",
            "message": f"Chat history cleared ({result.deleted_count} messages deleted)"
        }
