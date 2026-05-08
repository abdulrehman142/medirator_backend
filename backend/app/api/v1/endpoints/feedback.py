from fastapi import APIRouter, Depends

from app.core.deps import get_current_user
from app.db.mongo import get_database
from app.schemas.feedback import FeedbackCreate, FeedbackCreateInput, FeedbackPublic, FeedbackTarget, FeedbackUpdate
from app.schemas.user import UserPublic
from app.services.feedback_service import FeedbackService
from app.services.security_service import SecurityService

router = APIRouter()


@router.post("", response_model=FeedbackPublic)
async def create_feedback(payload: FeedbackCreateInput, current_user: UserPublic = Depends(get_current_user)):
    db = get_database()
    create_payload = FeedbackCreate(**payload.model_dump(), user_id=current_user.id, role=current_user.role.value)
    feedback = await FeedbackService(db).create(create_payload)
    await SecurityService(db).log_audit("feedback.create", actor_id=current_user.id, target_id=feedback.id)
    return feedback


@router.get("", response_model=list[FeedbackPublic])
async def list_feedback(target_type: FeedbackTarget | None = None, role: str | None = None):
    db = get_database()
    return await FeedbackService(db).list(role=role, target_type=target_type)


@router.patch("/{feedback_id}", response_model=FeedbackPublic)
async def update_feedback(
    feedback_id: str,
    payload: FeedbackUpdate,
    current_user: UserPublic = Depends(get_current_user),
):
    db = get_database()
    updated = await FeedbackService(db).update_own(feedback_id, current_user.id, payload)
    await SecurityService(db).log_audit("feedback.update", actor_id=current_user.id, target_id=feedback_id)
    return updated
