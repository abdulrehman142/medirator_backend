from fastapi import APIRouter

from app.api.v1.endpoints import admin, appointments, auth, chat, clinical, disease_prediction, feedback, ml, reports, security, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(appointments.router, prefix="/appointments", tags=["appointments"])
api_router.include_router(clinical.router, prefix="/clinical", tags=["clinical"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(security.router, prefix="/security", tags=["security"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(disease_prediction.router, prefix="/symptom-predictor", tags=["symptom-predictor"])
api_router.include_router(disease_prediction.router, prefix="/disease-prediction", tags=["disease-prediction"])
api_router.include_router(ml.router, prefix="/ml", tags=["ml"])
