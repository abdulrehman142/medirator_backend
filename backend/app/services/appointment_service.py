from datetime import UTC, datetime

from bson import ObjectId
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.appointment import AppointmentCreate, AppointmentPublic, AppointmentStatus, AppointmentUpdate
from app.services.user_service import UserService
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow


class AppointmentService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.user_service = UserService(db)

    @staticmethod
    def _format_scheduled_for_display(value: datetime) -> str:
        # Keep a single presentation format for clients that need a human-readable time.
        normalized = value if value.tzinfo else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).strftime("%b %d, %Y, %I:%M %p UTC")

    async def create(
        self,
        payload: AppointmentCreate,
        current_user_id: str | None = None,
        current_user_role: str | None = None,
    ) -> AppointmentPublic:
        patient_id = payload.patient_id or current_user_id
        if not patient_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="patient_id is required")

        if current_user_role == "patient" and payload.patient_id and payload.patient_id != current_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patients can only book for themselves")

        patient = await self.user_service.get_user_by_id(patient_id)
        doctor = await self.user_service.get_user_by_id(payload.doctor_id)
        if not patient or patient.get("role") != "patient":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid patient")
        if not doctor or doctor.get("role") != "doctor" or not doctor.get("is_active") or doctor.get("is_blocked"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid doctor")

        doc = {
            "patient_id": ObjectId(patient_id),
            "doctor_id": ObjectId(payload.doctor_id),
            "reason": payload.reason,
            "scheduled_for": payload.scheduled_for,
            "status": AppointmentStatus.SCHEDULED.value,
            "notes": None,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        res = await self.db.appointments.insert_one(doc)
        return await self.get_by_id(str(res.inserted_id))

    async def get_by_id(self, appointment_id: str) -> AppointmentPublic | None:
        doc = await self.db.appointments.find_one({"_id": ObjectId(appointment_id)})
        if not doc:
            return None
        doc = to_str_id(doc)
        return AppointmentPublic(
            id=doc["id"],
            patient_id=str(doc["patient_id"]),
            doctor_id=str(doc["doctor_id"]),
            reason=doc["reason"],
            scheduled_for=doc["scheduled_for"],
            scheduled_for_display=self._format_scheduled_for_display(doc["scheduled_for"]),
            status=doc["status"],
            notes=doc.get("notes"),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
        )

    async def update(
        self,
        appointment_id: str,
        payload: AppointmentUpdate,
        current_user_id: str | None = None,
        current_user_role: str | None = None,
    ) -> AppointmentPublic | None:
        appointment = await self.db.appointments.find_one({"_id": ObjectId(appointment_id)})
        if not appointment:
            return None

        if current_user_role == "patient" and current_user_id and str(appointment["patient_id"]) != current_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify this appointment")
        if current_user_role == "doctor" and current_user_id and str(appointment["doctor_id"]) != current_user_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify this appointment")

        set_doc = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
        if not set_doc:
            return await self.get_by_id(appointment_id)

        if "reason" in set_doc and current_user_role != "admin" and current_user_role != "patient":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only patients or admins can change reason")

        status_value = set_doc.get("status")
        if status_value in {AppointmentStatus.RESCHEDULED, AppointmentStatus.CANCELED, AppointmentStatus.COMPLETED}:
            set_doc["status"] = status_value.value

        set_doc["updated_at"] = utcnow()
        await self.db.appointments.update_one({"_id": ObjectId(appointment_id)}, {"$set": set_doc})
        return await self.get_by_id(appointment_id)

    async def list(self, user_id: str, role: str, start: datetime | None, end: datetime | None) -> list[AppointmentPublic]:
        query: dict = {}
        if role == "patient":
            query["patient_id"] = ObjectId(user_id)
        elif role == "doctor":
            query["doctor_id"] = ObjectId(user_id)
        elif role != "admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

        if start or end:
            range_query = {}
            if start:
                range_query["$gte"] = start
            if end:
                range_query["$lte"] = end
            query["scheduled_for"] = range_query

        cursor = self.db.appointments.find(query).sort("scheduled_for", 1)
        results = []
        async for doc in cursor:
            transformed = to_str_id(doc)
            results.append(
                AppointmentPublic(
                    id=transformed["id"],
                    patient_id=str(transformed["patient_id"]),
                    doctor_id=str(transformed["doctor_id"]),
                    reason=transformed["reason"],
                    scheduled_for=transformed["scheduled_for"],
                    scheduled_for_display=self._format_scheduled_for_display(transformed["scheduled_for"]),
                    status=transformed["status"],
                    notes=transformed.get("notes"),
                    created_at=transformed["created_at"],
                    updated_at=transformed["updated_at"],
                )
            )
        return results
