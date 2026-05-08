from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    await db.users.create_indexes(
        [
            IndexModel([("email", ASCENDING)], unique=True, name="users_email_unique"),
            IndexModel([("role", ASCENDING)], name="users_role_idx"),
            IndexModel([("created_at", DESCENDING)], name="users_created_desc_idx"),
        ]
    )

    await db.refresh_tokens.create_indexes(
        [
            IndexModel([("jti", ASCENDING)], unique=True, name="refresh_jti_unique"),
            IndexModel([("user_id", ASCENDING)], name="refresh_user_idx"),
            IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0, name="refresh_ttl_idx"),
        ]
    )

    await db.patients.create_indexes([IndexModel([("user_id", ASCENDING)], unique=True, name="patients_user_unique")])
    await db.doctors.create_indexes([IndexModel([("user_id", ASCENDING)], unique=True, name="doctors_user_unique")])

    await db.appointments.create_indexes(
        [
            IndexModel([("patient_id", ASCENDING), ("scheduled_for", ASCENDING)], name="appointments_patient_date_idx"),
            IndexModel([("doctor_id", ASCENDING), ("scheduled_for", ASCENDING)], name="appointments_doctor_date_idx"),
            IndexModel([("status", ASCENDING)], name="appointments_status_idx"),
        ]
    )

    await db.prescriptions.create_indexes([IndexModel([("patient_id", ASCENDING), ("created_at", DESCENDING)], name="prescriptions_patient_idx")])
    await db.clinical_notes.create_indexes([IndexModel([("patient_id", ASCENDING), ("created_at", DESCENDING)], name="notes_patient_idx")])
    await db.test_reports.create_indexes([IndexModel([("patient_id", ASCENDING), ("created_at", DESCENDING)], name="reports_patient_idx")])
    await db.documents.create_indexes([IndexModel([("owner_id", ASCENDING), ("created_at", DESCENDING)], name="documents_owner_idx")])
    await db.medications.create_indexes(
        [
            IndexModel([("patient_id", ASCENDING), ("status", ASCENDING), ("created_at", DESCENDING)], name="medications_patient_status_idx"),
            IndexModel([("doctor_id", ASCENDING), ("created_at", DESCENDING)], name="medications_doctor_idx"),
        ]
    )
    await db.feedback.create_indexes(
        [
            IndexModel([("role", ASCENDING), ("created_at", DESCENDING)], name="feedback_role_idx"),
            IndexModel([("target_type", ASCENDING), ("created_at", DESCENDING)], name="feedback_target_idx"),
        ]
    )
    await db.risk_assessments.create_indexes([IndexModel([("patient_id", ASCENDING), ("created_at", DESCENDING)], name="risk_patient_idx")])

    await db.security_events.create_indexes(
        [
            IndexModel([("event_type", ASCENDING), ("created_at", DESCENDING)], name="security_event_type_idx"),
            IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)], name="security_user_idx"),
        ]
    )

    await db.audit_logs.create_indexes(
        [
            IndexModel([("actor_id", ASCENDING), ("created_at", DESCENDING)], name="audit_actor_idx"),
            IndexModel([("action", ASCENDING), ("created_at", DESCENDING)], name="audit_action_idx"),
        ]
    )
