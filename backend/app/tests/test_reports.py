from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from bson import ObjectId
from fastapi import HTTPException

from app.schemas.report import ReportStatus, ReportStatusUpdate, TestReportCreate
from app.services.report_service import ReportService


def _report_doc(report_id: ObjectId, patient_id: ObjectId, doctor_id: ObjectId | None, metadata: dict[str, str]) -> dict:
    return {
        "_id": report_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "report_type": "Lab",
        "file_name": "result.pdf",
        "file_path": "/tmp/result.pdf",
        "storage_key": "result.pdf",
        "status": ReportStatus.UPLOADED.value,
        "metadata": metadata,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }


@pytest.mark.asyncio
async def test_create_report_persists_patient_metadata() -> None:
    db = AsyncMock()
    inserted_id = ObjectId()
    patient_id = ObjectId()
    doctor_id = ObjectId()

    db.test_reports.insert_one = AsyncMock(return_value=type("Result", (), {"inserted_id": inserted_id})())
    db.test_reports.find_one = AsyncMock(
        return_value=_report_doc(
            inserted_id,
            patient_id,
            doctor_id,
            {"source": "upload", "patient_id": str(patient_id), "doctor_id": str(doctor_id)},
        )
    )

    service = ReportService(db)
    payload = TestReportCreate(
        patient_id=str(patient_id),
        doctor_id=str(doctor_id),
        report_type="Lab",
        file_name="result.pdf",
        file_path="/tmp/result.pdf",
        storage_key="result.pdf",
        metadata={"source": "upload"},
    )

    report = await service.create(payload)

    assert report.id == str(inserted_id)
    inserted_doc = db.test_reports.insert_one.call_args.args[0]
    assert inserted_doc["metadata"]["patient_id"] == str(patient_id)
    assert inserted_doc["metadata"]["doctor_id"] == str(doctor_id)
    assert inserted_doc["metadata"]["source"] == "upload"


@pytest.mark.asyncio
async def test_update_status_merges_metadata_and_preserves_patient_scope() -> None:
    db = AsyncMock()
    report_id = ObjectId()
    patient_id = ObjectId()
    doctor_id = ObjectId()

    db.test_reports.find_one = AsyncMock(
        side_effect=[
            _report_doc(
                report_id,
                patient_id,
                doctor_id,
                {"patient_id": str(patient_id), "doctor_id": str(doctor_id), "original": "value"},
            ),
            _report_doc(
                report_id,
                patient_id,
                doctor_id,
                {
                    "patient_id": str(patient_id),
                    "doctor_id": str(doctor_id),
                    "original": "value",
                    "reviewed_by": "doctor-1",
                },
            ).copy() | {"status": ReportStatus.APPROVED.value},
        ]
    )
    db.test_reports.update_one = AsyncMock()
    db.appointments.find_one = AsyncMock(return_value=None)

    service = ReportService(db)
    updated = await service.update_status(
        str(report_id),
        ReportStatusUpdate(status=ReportStatus.APPROVED, metadata={"reviewed_by": "doctor-1"}),
        actor_id=str(doctor_id),
        actor_role="doctor",
    )

    assert updated is not None
    assert updated.status == ReportStatus.APPROVED
    assert updated.metadata["patient_id"] == str(patient_id)
    assert updated.metadata["doctor_id"] == str(doctor_id)
    assert updated.metadata["original"] == "value"
    assert updated.metadata["reviewed_by"] == "doctor-1"

    set_doc = db.test_reports.update_one.call_args.args[1]["$set"]
    assert set_doc["metadata"]["patient_id"] == str(patient_id)
    assert set_doc["metadata"]["doctor_id"] == str(doctor_id)
    assert set_doc["metadata"]["original"] == "value"
    assert set_doc["metadata"]["reviewed_by"] == "doctor-1"


@pytest.mark.asyncio
async def test_update_status_rejects_unrelated_doctor() -> None:
    db = AsyncMock()
    report_id = ObjectId()
    patient_id = ObjectId()
    report_doctor_id = ObjectId()
    actor_doctor_id = ObjectId()

    db.test_reports.find_one = AsyncMock(
        return_value=_report_doc(
            report_id,
            patient_id,
            report_doctor_id,
            {"patient_id": str(patient_id), "doctor_id": str(report_doctor_id)},
        )
    )
    db.test_reports.update_one = AsyncMock()
    db.appointments.find_one = AsyncMock(return_value=None)

    service = ReportService(db)

    with pytest.raises(HTTPException) as exc:
        await service.update_status(
            str(report_id),
            ReportStatusUpdate(status=ReportStatus.REJECTED),
            actor_id=str(actor_doctor_id),
            actor_role="doctor",
        )

    assert exc.value.status_code == 403
    db.test_reports.update_one.assert_not_awaited()