import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
import pytesseract

from app.models.collections import COLLECTIONS
from app.schemas.chat_history import ChatMessageResponse


class FileExtractionService:
    """Extracts and normalizes textual context from uploaded files."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    PDF_EXTENSIONS = {".pdf"}

    def __init__(self, max_chars_per_file: int = 6000):
        self.max_chars_per_file = max_chars_per_file

    def _truncate(self, text: str) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= self.max_chars_per_file:
            return normalized
        return f"{normalized[: self.max_chars_per_file]}..."

    def _extract_pdf_text(self, file_path: Path) -> str:
        try:
            reader = PdfReader(str(file_path))
            pages: List[str] = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return self._truncate("\n".join(pages))
        except Exception as exc:
            return f"[PDF extraction failed: {exc}]"

    def _extract_image_text(self, file_path: Path) -> str:
        try:
            image = Image.open(file_path)
            extracted = pytesseract.image_to_string(image)
            cleaned = self._truncate(extracted)
            return cleaned if cleaned else "[No readable text found in image.]"
        except Exception as exc:
            return f"[Image OCR failed: {exc}]"

    def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in self.PDF_EXTENSIONS:
            return self._extract_pdf_text(path)
        if suffix in self.IMAGE_EXTENSIONS:
            return self._extract_image_text(path)
        return "[Unsupported file type for text extraction.]"

    def process_uploaded_files(
        self, temp_files: List[str], original_files: Optional[List[Any]]
    ) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        files = original_files or []
        for idx, temp_path in enumerate(temp_files):
            incoming = files[idx] if idx < len(files) else None
            filename = getattr(incoming, "filename", Path(temp_path).name)
            content_type = getattr(incoming, "content_type", "unknown")
            extracted_text = self.extract_text(temp_path)
            metadata.append(
                {
                    "filename": filename,
                    "content_type": content_type or "unknown",
                    "extracted_text": extracted_text,
                }
            )
        return metadata


file_extraction_service_instance = FileExtractionService()


class ChatSessionService:
    """Manages multi-session chat functionality"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.sessions_collection = db[COLLECTIONS["chat_sessions"]]
        self.messages_collection = db[COLLECTIONS["chat_messages"]]
        self.attachments_collection = db[COLLECTIONS["chat_attachments"]]

    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user"""
        cursor = self.sessions_collection.find({"user_id": user_id, "is_deleted": False}).sort("created_at", -1)
        sessions = []
        async for session in cursor:
            # Count messages in this session
            message_count = await self.messages_collection.count_documents({"chat_session_id": session["_id"]})
            sessions.append({
                "id": session["_id"],
                "title": session.get("title", "Untitled Chat"),
                "created_at": session.get("created_at"),
                "updated_at": session.get("updated_at"),
                "message_count": message_count
            })
        return sessions

    async def create_session(self, user_id: str, title: str = "New Chat") -> Dict[str, Any]:
        """Create a new chat session"""
        session_id = uuid.uuid4().hex
        now = datetime.utcnow()
        
        session_doc = {
            "_id": session_id,
            "user_id": user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "is_deleted": False
        }
        
        await self.sessions_collection.insert_one(session_doc)
        
        return {
            "id": session_id,
            "title": title,
            "created_at": now
        }

    async def update_session_title(self, session_id: str, user_id: str, title: str) -> Dict[str, Any]:
        """Update chat session title"""
        result = await self.sessions_collection.update_one(
            {"_id": session_id, "user_id": user_id},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            raise ValueError("Chat session not found")
        
        return {
            "id": session_id,
            "title": title,
            "updated_at": datetime.utcnow()
        }

    async def delete_session(self, session_id: str, user_id: str) -> Dict[str, str]:
        """Delete a chat session and all its messages"""
        # Verify ownership
        session = await self.sessions_collection.find_one({"_id": session_id, "user_id": user_id})
        if not session:
            raise ValueError("Chat session not found")
        
        # Mark as deleted (soft delete)
        await self.sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
        )
        
        return {
            "status": "success",
            "message": "Chat session deleted"
        }

    async def save_message(
        self,
        session_id: str,
        user_id: str,
        role: str,  # "user" or "assistant"
        content: str,
        model_used: Optional[str] = None,
        actual_model_used: Optional[str] = None,
        confidence: Optional[float] = None,
        selection_reason: Optional[str] = None,
        files_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Save a message to a chat session"""
        message_id = uuid.uuid4().hex
        timestamp = datetime.utcnow()
        
        message_doc = {
            "_id": message_id,
            "chat_session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "model_used": model_used,
            "actual_model_used": actual_model_used,
            "confidence": confidence,
            "selection_reason": selection_reason,
            "files_metadata": files_metadata or [],
            "timestamp": timestamp
        }
        
        await self.messages_collection.insert_one(message_doc)
        
        # Update session's updated_at timestamp
        await self.sessions_collection.update_one(
            {"_id": session_id},
            {"$set": {"updated_at": timestamp}}
        )
        
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "timestamp": timestamp
        }

    async def get_session_messages(self, session_id: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all messages in a session (for context)"""
        cursor = self.messages_collection.find(
            {"chat_session_id": session_id, "user_id": user_id}
        ).sort("timestamp", 1).limit(limit)
        
        messages = []
        async for msg in cursor:
            messages.append({
                "id": msg["_id"],
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp"),
                "model_used": msg.get("model_used"),
                "actual_model_used": msg.get("actual_model_used"),
                "confidence": msg.get("confidence"),
                "selection_reason": msg.get("selection_reason"),
                "files": msg.get("files_metadata", [])
            })
        return messages

    async def save_attachment(
        self,
        message_id: str,
        filename: str,
        file_type: str,
        extracted_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save file attachment metadata"""
        attachment_id = uuid.uuid4().hex
        
        attachment_doc = {
            "_id": attachment_id,
            "message_id": message_id,
            "filename": filename,
            "file_type": file_type,
            "extracted_text": extracted_text,
            "uploaded_at": datetime.utcnow()
        }
        
        await self.attachments_collection.insert_one(attachment_doc)
        
        return {
            "id": attachment_id,
            "filename": filename,
            "extracted_text": extracted_text
        }
