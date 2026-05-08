from pydantic import BaseModel, Field

class AIQueryRequest(BaseModel):
    # This is what the user sends to the backend
    prompt: str = Field(..., example="What are the risks of high blood pressure?")
    max_tokens: int = Field(default=200, ge=1, le=500)

class AIQueryResponse(BaseModel):
    # This is what the backend sends back
    response: str
    status: str = "success"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model_preference: str | None = Field(default="gemini", description='"gemini" | "xrayas"')


class ChatResponse(BaseModel):
    reply: str
    model_used: str | None = None
    actual_model_used: str | None = None
    confidence: float | None = None
    selection_reason: str | None = None
    disclaimer: str | None = None