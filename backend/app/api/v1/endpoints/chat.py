from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from typing import List, Optional
import shutil
import os
import uuid
from datetime import datetime

from app.core.deps import oauth2_scheme, get_current_user
from app.db.mongo import get_database
from app.schemas.user import UserPublic
from app.schemas.ai import ChatRequest, ChatResponse
from app.services.ai_orchestrator import AIOrchestrator
from app.services.chat_session_service import ChatSessionService, file_extraction_service_instance
from app.vision import maybe_image_path

router = APIRouter()
ai_orchestrator = AIOrchestrator()


async def get_optional_user(request: Request) -> Optional[UserPublic]:
    """Get user from Bearer token if present"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    try:
        token = auth_header.split(" ")[1]
        user = await get_current_user(token)
        return user
    except Exception:
        return None


@router.get("/readiness")
async def chat_readiness():
    """Runtime health for Gemini + XRayAS components."""
    try:
        status = await ai_orchestrator.get_runtime_status()
        return {"status": "success", "data": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Readiness check failed: {e}")


@router.post("", response_model=ChatResponse)
async def chat_simple(payload: ChatRequest, request: Request):
    """
    Simple production-safe chat endpoint.
    - Works for the frontend route that previously hit POST /api/v1/chat/ (404)
    - Uses Gemini for chat and XRayAS for chest x-ray image analysis
    - Does not require session storage
    """
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to use MediBot chatbot.")

    model_pref = (payload.model_preference or "gemini") if hasattr(payload, "model_preference") else "gemini"
    result = await ai_orchestrator.process_interaction(
        text=payload.message, conversation_history=None, model_preference=model_pref
    )
    return ChatResponse(
        reply=result.get("answer", ""),
        model_used=result.get("model_used"),
        actual_model_used=result.get("actual_model_used"),
        confidence=result.get("confidence"),
        selection_reason=result.get("selection_reason"),
        disclaimer=result.get("disclaimer"),
    )


# ============================================================================
# Session Management Endpoints
# ============================================================================

@router.get("/sessions")
async def get_chat_sessions(request: Request):
    """Get all chat sessions for authenticated user"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to access chat history.")
    
    try:
        db = get_database()
        session_service = ChatSessionService(db)
        sessions = await session_service.get_user_sessions(user.id)
        
        return {
            "status": "success",
            "data": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions")
async def create_chat_session(request: Request):
    """Create a new chat session"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to create a chat.")
    
    try:
        # Parse JSON body
        try:
            body = await request.json()
        except:
            body = {}
        
        title = body.get("title", "New Chat") if isinstance(body, dict) else "New Chat"
        
        db = get_database()
        session_service = ChatSessionService(db)
        session = await session_service.create_session(user.id, title)
        
        return {
            "status": "success",
            "data": session
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}")
async def update_chat_session(session_id: str, request: Request):
    """Update chat session (rename title)"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to update chat.")
    
    try:
        body = await request.json()
        title = body.get("title")
        
        if not title:
            raise ValueError("Title is required")
        
        db = get_database()
        session_service = ChatSessionService(db)
        result = await session_service.update_session_title(session_id, user.id, title)
        
        return {
            "status": "success",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, request: Request):
    """Delete a chat session"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to delete chat.")
    
    try:
        db = get_database()
        session_service = ChatSessionService(db)
        result = await session_service.delete_session(session_id, user.id)
        
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, request: Request, limit: int = 50):
    """Get all messages from a specific chat session"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to access chat history.")
    
    try:
        db = get_database()
        session_service = ChatSessionService(db)
        
        # Verify session exists and belongs to user
        session = await db["chat_sessions"].find_one({"_id": session_id, "user_id": user.id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        messages = await session_service.get_session_messages(session_id, user.id, limit)
        
        return {
            "status": "success",
            "data": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_all_chat_history(request: Request):
    """Clear all chat history for authenticated user"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to clear chat history.")
    
    try:
        db = get_database()
        
        # Mark all user sessions as deleted
        result = await db["chat_sessions"].update_many(
            {"user_id": user.id},
            {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
        )
        
        return {
            "status": "success",
            "message": f"All chat history cleared ({result.modified_count} sessions)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Message Endpoints
# ============================================================================

@router.post("/message")
async def send_chat_message(
    request: Request,
    chat_id: str = Form(...),
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    model_preference: str = Form("gemini")
):
    """Send a message to a chat session with optional file attachments"""
    user = await get_optional_user(request)
    if not user:
        return {"error": "Please log in to use MediBot chatbot."}
    
    if not message or message.strip() == "":
        raise HTTPException(status_code=400, detail="Message content is required")
    
    db = get_database()
    session_service = ChatSessionService(db)
    temp_files = []
    
    try:
        # Step 1: Process uploaded files
        files_metadata = []
        for file in files or []:
            if file and file.filename:
                temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                temp_files.append(temp_path)

        files_metadata = file_extraction_service_instance.process_uploaded_files(
            temp_files=temp_files,
            original_files=files,
        )
        
        # Step 2: Save user message to DB
        user_message_result = await session_service.save_message(
            session_id=chat_id,
            user_id=user.id,
            role="user",
            content=message,
            files_metadata=files_metadata
        )
        
        # Step 3: Get conversation history for context
        history = await session_service.get_session_messages(chat_id, user.id, limit=20)
        
        # Step 4: Call AI Orchestrator with context — only real image uploads go to XRayAS when selected
        image_path = None
        for temp_path, upload in zip(temp_files or [], files or []):
            if upload is None or not upload.filename:
                continue
            cand, ok = maybe_image_path(upload.filename, temp_path)
            if ok and cand:
                image_path = cand
                break
        ai_response = await ai_orchestrator.process_interaction(
            text=message,
            image_path=image_path,
            conversation_history=history,
            uploaded_file_contexts=files_metadata,
            model_preference=(model_preference or "gemini"),
        )
        
        # Step 5: Save AI response to DB
        assistant_message_result = await session_service.save_message(
            session_id=chat_id,
            user_id=user.id,
            role="assistant",
            content=ai_response["answer"],
            model_used=ai_response.get("model_used"),
            actual_model_used=ai_response.get("actual_model_used"),
            confidence=ai_response.get("confidence"),
            selection_reason=ai_response.get("selection_reason"),
        )
        
        # Step 6: Save attachments metadata
        processed_files = []
        for metadata in files_metadata:
            attachment = await session_service.save_attachment(
                message_id=user_message_result["id"],
                filename=metadata["filename"],
                file_type=str(metadata.get("content_type") or "unknown"),
                extracted_text=metadata.get("extracted_text"),
            )
            processed_files.append({
                "filename": metadata["filename"],
                "extracted_text": attachment.get("extracted_text")
            })
        
        # Step 7: Cleanup temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return {
            "status": "success",
            "data": {
                "message_id": assistant_message_result["id"],
                "user_message": message,
                "assistant_reply": ai_response["answer"],
                "model_used": ai_response.get("model_used"),
                "actual_model_used": ai_response.get("actual_model_used"),
                "confidence": ai_response.get("confidence"),
                "selection_reason": ai_response.get("selection_reason"),
                "files_processed": processed_files,
                "timestamp": assistant_message_result["timestamp"]
            }
        }
        
    except Exception as e:
        # Cleanup on error
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legacy Endpoint (for backward compatibility during migration)
# ============================================================================

@router.get("/history")
async def get_chat_history_legacy(request: Request, limit: int = 50):
    """[DEPRECATED] Use GET /sessions instead"""
    user = await get_optional_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to use MediBot chatbot.")
    
    # For backward compatibility, return the most recent session's messages
    try:
        db = get_database()
        session_service = ChatSessionService(db)
        sessions = await session_service.get_user_sessions(user.id)
        
        if not sessions:
            return []
        
        # Get messages from most recent session
        most_recent_session_id = sessions[0]["id"]
        messages = await session_service.get_session_messages(most_recent_session_id, user.id, limit)
        
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
