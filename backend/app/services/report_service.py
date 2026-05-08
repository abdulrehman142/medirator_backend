from bson import ObjectId
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List

from app.schemas.report import ReportStatus, ReportStatusUpdate, TestReportCreate, TestReportPublic
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow


class ReportService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    @staticmethod
    def _build_metadata(payload: TestReportCreate) -> dict[str, str]:
        metadata = dict(payload.metadata)
        metadata["patient_id"] = payload.patient_id
        if payload.doctor_id:
            metadata["doctor_id"] = payload.doctor_id
        return metadata

    @staticmethod
    def _to_public(doc: dict) -> TestReportPublic:
        d = to_str_id(doc)
        return TestReportPublic(
            id=d["id"],
            patient_id=str(d["patient_id"]),
            doctor_id=str(d["doctor_id"]) if d.get("doctor_id") else None,
            report_type=d["report_type"],
            file_name=d["file_name"],
            file_path=d["file_path"],
            storage_key=d.get("storage_key"),
            status=ReportStatus(d.get("status", ReportStatus.UPLOADED.value)),
            metadata=d["metadata"],
            created_at=d["created_at"],
        )

    async def create(self, payload: TestReportCreate) -> TestReportPublic:
        metadata = self._build_metadata(payload)
        res = await self.db.test_reports.insert_one(
            {
                "patient_id": ObjectId(payload.patient_id),
                "doctor_id": ObjectId(payload.doctor_id) if payload.doctor_id else None,
                "report_type": payload.report_type,
                "file_name": payload.file_name,
                "file_path": payload.file_path,
                "storage_key": payload.storage_key or payload.file_path,
                "status": payload.status.value,
                "metadata": metadata,
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
        )
        return await self.get_by_id(str(res.inserted_id))

    async def get_by_id(self, report_id: str) -> TestReportPublic | None:
        doc = await self.db.test_reports.find_one({"_id": ObjectId(report_id)})
        if not doc:
            return None
        return self._to_public(doc)

    async def list(self, patient_id: str | None = None, doctor_id: str | None = None, report_type: str | None = None) -> List[TestReportPublic]:
        query: dict = {}
        if patient_id:
            query["patient_id"] = ObjectId(patient_id)
        if doctor_id:
            query["doctor_id"] = ObjectId(doctor_id)
        if report_type:
            query["report_type"] = report_type

        cursor = self.db.test_reports.find(query).sort("created_at", -1)
        items = []
        async for doc in cursor:
            items.append(self._to_public(doc))
        return items

    async def list_for_role(self, user_id: str, role: str, patient_id: str | None = None, report_type: str | None = None) -> List[TestReportPublic]:
        if role == "admin":
            return await self.list(patient_id=patient_id, report_type=report_type)
        if role == "patient":
            return await self.list(patient_id=user_id, report_type=report_type)
        if role == "doctor":
            patient_ids = await self.db.appointments.distinct("patient_id", {"doctor_id": ObjectId(user_id)})
            query = {"$or": [{"doctor_id": ObjectId(user_id)}]}
            if patient_ids:
                query["$or"].append({"patient_id": {"$in": patient_ids}})
            if report_type:
                query["report_type"] = report_type
            cursor = self.db.test_reports.find(query).sort("created_at", -1)
            items: list[TestReportPublic] = []
            async for doc in cursor:
                items.append(self._to_public(doc))
            return items
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

    async def update_status(self, report_id: str, payload: ReportStatusUpdate, actor_id: str, actor_role: str) -> TestReportPublic | None:
        if actor_role not in {"doctor", "admin"}:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

        doc = await self.db.test_reports.find_one({"_id": ObjectId(report_id)})
        if not doc:
            return None

        if actor_role == "doctor":
            report_doctor_id = str(doc["doctor_id"]) if doc.get("doctor_id") else None
            if report_doctor_id not in {None, actor_id}:
                relation = await self.db.appointments.find_one({"doctor_id": ObjectId(actor_id), "patient_id": doc["patient_id"]})
                if not relation:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
            elif report_doctor_id is None:
                relation = await self.db.appointments.find_one({"doctor_id": ObjectId(actor_id), "patient_id": doc["patient_id"]})
                if not relation:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

        set_doc = {"updated_at": utcnow()}
        if payload.status is not None:
            set_doc["status"] = payload.status.value

        metadata = dict(doc.get("metadata") or {})
        if payload.metadata is not None:
            metadata.update(payload.metadata)
        metadata["patient_id"] = str(doc["patient_id"])
        if doc.get("doctor_id"):
            metadata["doctor_id"] = str(doc["doctor_id"])
        set_doc["metadata"] = metadata

        await self.db.test_reports.update_one({"_id": ObjectId(report_id)}, {"$set": set_doc})
        return await self.get_by_id(report_id)

    async def delete_report(self, report_id: str) -> bool:
        result = await self.db.test_reports.delete_one({"_id": ObjectId(report_id)})
        return result.deleted_count > 0
