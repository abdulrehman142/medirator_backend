from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas.clinical import ClinicalNoteCreate, MedicalHistoryItem, MedicationCreate, MedicationPublic, MedicationStatus, MedicationUpdate, PrescriptionCreate, RiskAssessmentCreate, RiskAssessmentPublic, TimelineRecord
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow


class ClinicalService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def upsert_medical_history(self, payload: MedicalHistoryItem) -> None:
        await self.db.patients.update_one(
            {"user_id": ObjectId(payload.patient_id)},
            {
                "$set": {
                    "diagnosis": payload.diagnosis,
                    "chronic_conditions": payload.chronic_conditions,
                    "allergies": payload.allergies,
                    "updated_at": utcnow(),
                },
                "$setOnInsert": {"created_at": utcnow()},
            },
            upsert=True,
        )

    async def get_medical_history(self, patient_id: str) -> dict | None:
        doc = await self.db.patients.find_one({"user_id": ObjectId(patient_id)})
        return to_str_id(doc) if doc else None

    async def add_note(self, payload: ClinicalNoteCreate) -> str:
        res = await self.db.clinical_notes.insert_one(
            {
                "patient_id": ObjectId(payload.patient_id),
                "doctor_id": ObjectId(payload.doctor_id),
                "note": payload.note,
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
        )
        return str(res.inserted_id)

    async def delete_note(self, note_id: str) -> bool:
        result = await self.db.clinical_notes.delete_one({"_id": ObjectId(note_id)})
        return result.deleted_count > 0

    async def add_prescription(self, payload: PrescriptionCreate) -> str:
        res = await self.db.prescriptions.insert_one(
            {
                "patient_id": ObjectId(payload.patient_id),
                "doctor_id": ObjectId(payload.doctor_id),
                "medication": payload.medication,
                "dosage": payload.dosage,
                "instructions": payload.instructions,
                "created_at": utcnow(),
                "updated_at": utcnow(),
            }
        )
        return str(res.inserted_id)

    async def add_medication(self, payload: MedicationCreate) -> MedicationPublic:
        now = utcnow()
        res = await self.db.medications.insert_one(
            {
                "patient_id": ObjectId(payload.patient_id),
                "doctor_id": ObjectId(payload.doctor_id),
                "medication_name": payload.medication_name,
                "dosage": payload.dosage,
                "instructions": payload.instructions,
                "status": payload.status.value,
                "start_date": payload.start_date or now,
                "end_date": payload.end_date,
                "created_at": now,
                "updated_at": now,
            }
        )
        return await self.get_medication(str(res.inserted_id))

    async def get_medication(self, medication_id: str) -> MedicationPublic | None:
        doc = await self.db.medications.find_one({"_id": ObjectId(medication_id)})
        if not doc:
            return None
        d = to_str_id(doc)
        return MedicationPublic(
            id=d["id"],
            patient_id=str(d["patient_id"]),
            doctor_id=str(d["doctor_id"]),
            medication_name=d["medication_name"],
            dosage=d["dosage"],
            instructions=d["instructions"],
            status=MedicationStatus(d.get("status", MedicationStatus.CURRENT.value)),
            start_date=d.get("start_date"),
            end_date=d.get("end_date"),
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )

    async def list_medications(self, patient_id: str, status: MedicationStatus | None = None) -> list[MedicationPublic]:
        patient_oid = ObjectId(patient_id)
        now = utcnow()

        # Automatically move completed courses to past if end_date has elapsed.
        await self.db.medications.update_many(
            {
                "patient_id": patient_oid,
                "status": MedicationStatus.CURRENT.value,
                "end_date": {"$ne": None, "$lte": now},
            },
            {"$set": {"status": MedicationStatus.PAST.value, "updated_at": now}},
        )

        query = {"patient_id": patient_oid}
        if status:
            query["status"] = status.value
        cursor = self.db.medications.find(query).sort("created_at", -1)
        items: list[MedicationPublic] = []
        async for doc in cursor:
            d = to_str_id(doc)
            items.append(
                MedicationPublic(
                    id=d["id"],
                    patient_id=str(d["patient_id"]),
                    doctor_id=str(d["doctor_id"]),
                    medication_name=d["medication_name"],
                    dosage=d["dosage"],
                    instructions=d["instructions"],
                    status=MedicationStatus(d.get("status", MedicationStatus.CURRENT.value)),
                    start_date=d.get("start_date"),
                    end_date=d.get("end_date"),
                    created_at=d["created_at"],
                    updated_at=d["updated_at"],
                )
            )
        return items

    async def update_medication_status(self, medication_id: str, payload: MedicationUpdate) -> MedicationPublic | None:
        update_doc: dict = {"updated_at": utcnow()}
        if payload.status is not None:
            update_doc["status"] = payload.status.value
            if payload.status == MedicationStatus.CURRENT:
                # Restoring a medication should make it active again.
                # If end_date remains in the past, list_medications auto-demotes it back to past.
                update_doc["end_date"] = None
            if payload.status in {MedicationStatus.PAST, MedicationStatus.INACTIVE} and payload.end_date is None:
                update_doc["end_date"] = utcnow()
        if payload.dosage is not None:
            update_doc["dosage"] = payload.dosage
        if payload.instructions is not None:
            update_doc["instructions"] = payload.instructions
        if payload.end_date is not None:
            update_doc["end_date"] = payload.end_date
        await self.db.medications.update_one({"_id": ObjectId(medication_id)}, {"$set": update_doc})
        return await self.get_medication(medication_id)

    async def add_risk_assessment(self, payload: RiskAssessmentCreate) -> RiskAssessmentPublic:
        now = utcnow()
        res = await self.db.risk_assessments.insert_one(
            {
                "patient_id": ObjectId(payload.patient_id),
                "score": payload.score,
                "summary": payload.summary,
                "metadata": payload.metadata,
                "created_at": now,
            }
        )
        return await self.get_risk_assessment(str(res.inserted_id))

    async def get_risk_assessment(self, assessment_id: str) -> RiskAssessmentPublic | None:
        doc = await self.db.risk_assessments.find_one({"_id": ObjectId(assessment_id)})
        if not doc:
            return None
        d = to_str_id(doc)
        return RiskAssessmentPublic(
            id=d["id"],
            patient_id=str(d["patient_id"]),
            score=d["score"],
            summary=d["summary"],
            metadata=d.get("metadata", {}),
            created_at=d["created_at"],
        )

    async def list_risk_assessments(self, patient_id: str) -> list[RiskAssessmentPublic]:
        cursor = self.db.risk_assessments.find({"patient_id": ObjectId(patient_id)}).sort("created_at", -1)
        items: list[RiskAssessmentPublic] = []
        async for doc in cursor:
            d = to_str_id(doc)
            items.append(
                RiskAssessmentPublic(
                    id=d["id"],
                    patient_id=str(d["patient_id"]),
                    score=d["score"],
                    summary=d["summary"],
                    metadata=d.get("metadata", {}),
                    created_at=d["created_at"],
                )
            )
        return items

    async def timeline(self, patient_id: str) -> list[TimelineRecord]:
        pid = ObjectId(patient_id)
        records: list[TimelineRecord] = []

        async for note in self.db.clinical_notes.find({"patient_id": pid}):
            note = to_str_id(note)
            records.append(
                TimelineRecord(
                    id=note["id"],
                    record_type="clinical_note",
                    patient_id=patient_id,
                    doctor_id=str(note["doctor_id"]),
                    summary=note["note"],
                    created_at=note["created_at"],
                )
            )

        async for rx in self.db.prescriptions.find({"patient_id": pid}):
            rx = to_str_id(rx)
            records.append(
                TimelineRecord(
                    id=rx["id"],
                    record_type="prescription",
                    patient_id=patient_id,
                    doctor_id=str(rx["doctor_id"]),
                    summary=f"{rx['medication']} - {rx['dosage']}",
                    created_at=rx["created_at"],
                )
            )

        async for report in self.db.test_reports.find({"patient_id": pid}):
            report = to_str_id(report)
            records.append(
                TimelineRecord(
                    id=report["id"],
                    record_type="test_report",
                    patient_id=patient_id,
                    doctor_id=str(report["doctor_id"]),
                    summary=f"{report['report_type']} ({report['file_name']})",
                    created_at=report["created_at"],
                )
            )

        async for medication in self.db.medications.find({"patient_id": pid}):
            med = to_str_id(medication)
            records.append(
                TimelineRecord(
                    id=med["id"],
                    record_type="medication",
                    patient_id=patient_id,
                    doctor_id=str(med["doctor_id"]),
                    summary=f"{med['medication_name']} - {med['dosage']} ({med.get('status', 'current')})",
                    created_at=med["created_at"],
                )
            )

        async for risk in self.db.risk_assessments.find({"patient_id": pid}):
            assessment = to_str_id(risk)
            records.append(
                TimelineRecord(
                    id=assessment["id"],
                    record_type="risk_assessment",
                    patient_id=patient_id,
                    doctor_id=None,
                    summary=f"Risk score {assessment['score']}: {assessment['summary']}",
                    created_at=assessment["created_at"],
                )
            )

        return sorted(records, key=lambda item: item.created_at, reverse=True)

    async def unified_records(self, patient_id: str) -> dict:
        history = await self.get_medical_history(patient_id)
        return {
            "family_history": history,
            "medical_history": history,
            "appointments": [item.model_dump() for item in await self.list_appointments(patient_id)],
            "reports": [item.model_dump() for item in await self.list_reports(patient_id)],
            "prescriptions": await self.list_prescriptions(patient_id),
            "notes": await self.list_notes(patient_id),
            "medications_current": [item.model_dump() for item in await self.list_medications(patient_id, MedicationStatus.CURRENT)],
            "medications_past": [item.model_dump() for item in await self.list_medications(patient_id, MedicationStatus.PAST)],
            "risk_assessments": [item.model_dump() for item in await self.list_risk_assessments(patient_id)],
            "timeline": [item.model_dump() for item in await self.timeline(patient_id)],
        }

    async def list_appointments(self, patient_id: str):
        from app.services.appointment_service import AppointmentService

        return await AppointmentService(self.db).list(patient_id, "patient", None, None)

    async def list_reports(self, patient_id: str):
        from app.services.report_service import ReportService

        return await ReportService(self.db).list(patient_id=patient_id)

    async def list_prescriptions(self, patient_id: str):
        cursor = self.db.prescriptions.find({"patient_id": ObjectId(patient_id)}).sort("created_at", -1)
        items = []
        async for rx in cursor:
            rx = to_str_id(rx)
            items.append({**rx, "doctor_id": str(rx["doctor_id"])})
        return items

    async def list_notes(self, patient_id: str):
        cursor = self.db.clinical_notes.find({"patient_id": ObjectId(patient_id)}).sort("created_at", -1)
        items = []
        async for note in cursor:
            note = to_str_id(note)
            items.append({**note, "doctor_id": str(note["doctor_id"])})
        return items

    async def compute_risk_score(self, patient_id: str) -> RiskAssessmentPublic:
        appointments = await self.db.appointments.count_documents({"patient_id": ObjectId(patient_id)})
        notes = await self.db.clinical_notes.count_documents({"patient_id": ObjectId(patient_id)})
        prescriptions = await self.db.prescriptions.count_documents({"patient_id": ObjectId(patient_id)})
        reports = await self.db.test_reports.count_documents({"patient_id": ObjectId(patient_id)})
        score = float(min(100, appointments * 5 + notes * 10 + prescriptions * 8 + reports * 6))
        summary = "Risk score computed from patient activity and clinical history"
        return await self.add_risk_assessment(RiskAssessmentCreate(patient_id=patient_id, score=score, summary=summary, metadata={"appointments": str(appointments), "notes": str(notes), "prescriptions": str(prescriptions), "reports": str(reports)}))
