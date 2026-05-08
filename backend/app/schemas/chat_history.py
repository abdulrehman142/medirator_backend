from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any

class ChatMessageBase(BaseModel):
    user_id: str
    session_id: str
    input_type: str = Field(default="text", description="text, voice, image, file")
    user_message: str
    files_metadata: Optional[List[Dict[str, Any]]] = None

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageResponse(BaseModel):
    id: str
    user_message: str
    answer: str
    confidence: Optional[float] = None
    model_used: str
    disclaimer: str = "This is not a medical diagnosis."
    timestamp: datetime

class ChatMessageDB(ChatMessageBase):
    id: str = Field(alias="_id")
    ai_response: str
    confidence: Optional[float] = None
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
