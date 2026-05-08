from datetime import UTC, date, timedelta

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.utils.id_formatter import format_patient_id, format_doctor_id, decode_formatted_id
from app.schemas.admin import (
    DashboardMetrics,
    DoctorAdminItem,
    DoctorAdminPatchRequest,
    DoctorManagementRole,
    DoctorManagementStatus,
    DoctorVerificationStatus,
    PatientAdminItem,
    PatientAdminPatchRequest,
    PatientManagementStatus,
)
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow
from app.services.user_service import UserService


class AdminService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def dashboard_metrics(self) -> DashboardMetrics:
        total_users = await self.db.users.count_documents({})
        total_appointments = await self.db.appointments.count_documents({})
        active_doctors = await self.db.users.count_documents({"role": "doctor", "is_active": True, "is_blocked": False})
        return DashboardMetrics(
            total_users=total_users,
            total_appointments=total_appointments,
            active_doctors=active_doctors,
        )

    async def basic_insights(self) -> dict:
        completed_appointments = await self.db.appointments.count_documents({"status": "completed"})
        canceled_appointments = await self.db.appointments.count_documents({"status": "canceled"})
        failed_logins = await self.db.security_events.count_documents({"event_type": "failed_login"})
        total_reports = await self.db.test_reports.count_documents({})
        return {
            "completed_appointments": completed_appointments,
            "canceled_appointments": canceled_appointments,
            "failed_logins": failed_logins,
            "total_reports": total_reports,
        }

    async def analytics_snapshot(self) -> dict:
        now = utcnow()
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)

        total_doctors = await self.db.users.count_documents({"role": "doctor"})
        total_patients = await self.db.users.count_documents({"role": "patient"})
        active_doctors = await self.db.users.count_documents({"role": "doctor", "is_active": True, "is_blocked": False})
        total_appointments = await self.db.appointments.count_documents({})
        total_completed_appointments = await self.db.appointments.count_documents({"status": "completed"})

        # Day-wise counts for chart widgets.
        patient_growth: list[dict] = []
        appointment_trends: list[dict] = []
        completed_appointment_trends: list[dict] = []
        for offset in range(6, -1, -1):
            day_start = (now - timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            new_patients = await self.db.users.count_documents(
                {"role": "patient", "created_at": {"$gte": day_start, "$lt": day_end}}
            )
            appointments = await self.db.appointments.count_documents(
                {"created_at": {"$gte": day_start, "$lt": day_end}}
            )
            # Use updated_at for completed appointments so completion date is respected
            completed_appointments = await self.db.appointments.count_documents(
                {"status": "completed", "updated_at": {"$gte": day_start, "$lt": day_end}}
            )
            patient_growth.append({"label": day_start.strftime("%a"), "value": new_patients})
            appointment_trends.append({"label": day_start.strftime("%a"), "value": appointments})
            completed_appointment_trends.append({"label": day_start.strftime("%a"), "value": completed_appointments})

        # Peak usage by appointment hour.
        pipeline_peak = [
            {"$match": {"created_at": {"$gte": seven_days_ago}}},
            {"$group": {"_id": {"$hour": "$created_at"}, "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 3},
        ]
        peak_usage_raw = await self.db.appointments.aggregate(pipeline_peak).to_list(length=3)
        peak_usage = [
            {
                "hour": int(item.get("_id", 0)),
                "label": f"{int(item.get('_id', 0)):02d}:00 - {((int(item.get('_id', 0)) + 1) % 24):02d}:00",
                "count": int(item.get("count", 0)),
            }
            for item in peak_usage_raw
        ]

        # Most active doctors (completed appointments).
        pipeline_active_doctors = [
            {
                "$match": {
                    "status": "completed",
                    # Use updated_at so we count appointments completed within the window
                    "updated_at": {"$gte": thirty_days_ago},
                }
            },
            {"$group": {"_id": "$doctor_id", "completed_appointments": {"$sum": 1}}},
            {"$sort": {"completed_appointments": -1}},
            {"$limit": 5},
        ]
        active_doctors_raw = await self.db.appointments.aggregate(pipeline_active_doctors).to_list(length=5)
        most_active_doctors: list[dict] = []
        for item in active_doctors_raw:
            doctor_user = await self.db.users.find_one({"_id": item["_id"]})
            doctor_name = (doctor_user or {}).get("full_name", "Unknown doctor")
            most_active_doctors.append(
                {
                    "doctor_id": format_doctor_id(str(item["_id"])),
                    "name": doctor_name,
                    "completed_appointments": int(item.get("completed_appointments", 0)),
                }
            )

        # Recent activity and alerts are derived from actual collections.
        recent_activity: list[str] = []
        new_users_24h = await self.db.users.count_documents({"created_at": {"$gte": now - timedelta(hours=24)}})
        recent_reports_24h = await self.db.test_reports.count_documents({"created_at": {"$gte": now - timedelta(hours=24)}})
        pending_doctor_verifications = await self.db.doctors.count_documents({"verification": {"$in": ["Pending", "pending"]}})
        recent_activity.append(f"{new_users_24h} new accounts registered in the last 24 hours.")
        recent_activity.append(f"{recent_reports_24h} test reports uploaded in the last 24 hours.")
        recent_activity.append(f"{pending_doctor_verifications} doctor accounts are pending verification.")

        alerts: list[str] = []
        failed_logins_24h = await self.db.security_events.count_documents(
            {"event_type": "failed_login", "created_at": {"$gte": now - timedelta(hours=24)}}
        )
        if failed_logins_24h > 0:
            alerts.append(f"{failed_logins_24h} failed login attempts detected in the last 24 hours.")
        if pending_doctor_verifications > 0:
            alerts.append(f"{pending_doctor_verifications} doctors are waiting for approval.")
        if not alerts:
            alerts.append("No critical alerts right now.")

        return {
            "totals": {
                "total_doctors": total_doctors,
                "total_patients": total_patients,
                "active_doctors": active_doctors,
                "total_appointments": total_appointments,
                "total_completed_appointments": total_completed_appointments,
            },
            "patient_growth": patient_growth,
            "appointment_trends": appointment_trends,
            "completed_appointment_trends": completed_appointment_trends,
            "peak_usage_times": peak_usage,
            "most_active_doctors": most_active_doctors,
            "recent_activity": recent_activity,
            "alerts": alerts,
        }

    @staticmethod
    def _to_object_id(value: str, *, field_name: str) -> ObjectId:
        try:
            return ObjectId(value)
        except (InvalidId, TypeError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {field_name}") from exc

    @staticmethod
    def _doctor_role_from_profile(profile: dict | None) -> DoctorManagementRole:
        role_value = (profile or {}).get("admin_role")
        if isinstance(role_value, str):
            normalized = role_value.strip().lower()
            if normalized == "senior doctor":
                return DoctorManagementRole.SENIOR_DOCTOR
            if normalized == "supervisor":
                return DoctorManagementRole.SUPERVISOR
        return DoctorManagementRole.DOCTOR

    @staticmethod
    def _doctor_status_from_user(user: dict) -> DoctorManagementStatus:
        is_active = bool(user.get("is_active", True))
        is_blocked = bool(user.get("is_blocked", False))
        return DoctorManagementStatus.ACTIVE if is_active and not is_blocked else DoctorManagementStatus.SUSPENDED

    @staticmethod
    def _doctor_verification_from_profile(profile: dict | None) -> DoctorVerificationStatus:
        verification = (profile or {}).get("verification")
        if isinstance(verification, str):
            normalized = verification.strip().lower()
            if normalized == "approved":
                return DoctorVerificationStatus.APPROVED
            if normalized == "rejected":
                return DoctorVerificationStatus.REJECTED
            if normalized == "pending":
                return DoctorVerificationStatus.PENDING

        if bool((profile or {}).get("is_verified")):
            return DoctorVerificationStatus.APPROVED
        return DoctorVerificationStatus.PENDING

    @staticmethod
    def _calculate_age(dob) -> int | None:
        """Calculate age from date_of_birth (handles date, datetime, and string)."""
        if not dob:
            return None
        
        try:
            # Convert to date if it's a datetime
            if hasattr(dob, 'date'):
                dob = dob.date()
            elif isinstance(dob, str):
                from datetime import datetime as dt
                dob = dt.fromisoformat(dob).date()
            
            today = date.today()
            calculated_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return max(0, calculated_age)  # Ensure age is not negative
        except (AttributeError, ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_text_list(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(cleaned) if cleaned else None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return str(value)

    async def _patient_status_from_user_activity(self, user_doc: dict) -> PatientManagementStatus:
        is_active = bool(user_doc.get("is_active", True))
        is_blocked = bool(user_doc.get("is_blocked", False))
        if not is_active or is_blocked:
            return PatientManagementStatus.INACTIVE

        # "Active" means currently logged in with at least one valid non-revoked refresh token.
        active_session_count = await self.db.refresh_tokens.count_documents(
            {
                "user_id": user_doc["_id"],
                "revoked": False,
                "expires_at": {"$gt": utcnow()},
            }
        )
        return PatientManagementStatus.ACTIVE if active_session_count > 0 else PatientManagementStatus.INACTIVE

    async def _resolve_doctor_user(self, doctor_id: str) -> tuple[ObjectId, dict]:
        # Direct ObjectId path (for raw IDs).
        try:
            object_id = ObjectId(doctor_id)
            doctor_user = await self.db.users.find_one({"_id": object_id, "role": "doctor"})
            if doctor_user:
                return object_id, doctor_user
        except (InvalidId, TypeError):
            pass

        # Formatted ID path (DR-XXXX-XXXX).
        cursor = self.db.users.find({"role": "doctor"})

    async def recent_completed_appointments(self, limit: int = 10) -> list[dict]:
        """Return recent completed appointments with optional resolved patient/doctor names."""
        results: list[dict] = []
        cursor = self.db.appointments.find({"status": "completed"}).sort("updated_at", -1).limit(limit)
        async for appt in cursor:
            patient_name = None
            doctor_name = None
            patient_id = appt.get("patient_id")
            doctor_id = appt.get("doctor_id")
            # Try to resolve names (support both string and ObjectId stored refs)
            try:
                from bson import ObjectId as _OID

                if patient_id:
                    try:
                        patient_obj = await self.db.users.find_one({"_id": _OID(patient_id)})
                    except Exception:
                        patient_obj = await self.db.users.find_one({"_id": patient_id})
                    if patient_obj:
                        patient_name = patient_obj.get("full_name")
                if doctor_id:
                    try:
                        doctor_obj = await self.db.users.find_one({"_id": _OID(doctor_id)})
                    except Exception:
                        doctor_obj = await self.db.users.find_one({"_id": doctor_id})
                    if doctor_obj:
                        doctor_name = doctor_obj.get("full_name")
            except Exception:
                # best-effort only
                pass

            results.append(
                {
                    "id": str(appt.get("_id")),
                    "patient_id": str(patient_id) if patient_id is not None else None,
                    "patient_name": patient_name,
                    "doctor_id": str(doctor_id) if doctor_id is not None else None,
                    "doctor_name": doctor_name,
                    "scheduled_for": str(appt.get("scheduled_for")) if appt.get("scheduled_for") is not None else None,
                    "updated_at": str(appt.get("updated_at")) if appt.get("updated_at") is not None else None,
                    "status": appt.get("status"),
                }
            )

        return results
        async for user in cursor:
            item = await self._build_doctor_admin_item(user)
            if item.id == doctor_id:
                return user["_id"], user

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Doctor not found")

    async def _build_doctor_admin_item(self, user_doc: dict) -> DoctorAdminItem:
        user = to_str_id(user_doc)
        doctor_doc = await self.db.doctors.find_one({"user_id": user_doc["_id"]})
        profile_data = to_str_id(doctor_doc) if doctor_doc else {}

        # Use doctor profile ID if available, otherwise use user ID, and format it
        doctor_id = profile_data.get("id") or user["id"]
        formatted_id = format_doctor_id(doctor_id)

        return DoctorAdminItem(
            id=formatted_id,
            name=user.get("full_name", ""),
            specialization=profile_data.get("specialization"),
            status=self._doctor_status_from_user(user),
            verification=self._doctor_verification_from_profile(profile_data),
            role=self._doctor_role_from_profile(profile_data),
        )

    async def _build_patient_admin_item(self, user_doc: dict) -> PatientAdminItem:
        user = to_str_id(user_doc)
        patient_doc = await self.db.patients.find_one({"user_id": user_doc["_id"]})
        profile_data = to_str_id(patient_doc) if patient_doc else {}

        # Use patient profile ID as the primary ID, fallback to user ID if no profile
        patient_id = profile_data.get("id") or user["id"]
        formatted_id = format_patient_id(patient_id)

        # Extract age from various possible field names in patient profile
        age = None
        for age_field in ["age", "patient_age", "years_of_age", "years_old"]:
            if age_field in profile_data and profile_data[age_field] is not None:
                try:
                    age = int(profile_data[age_field])
                    if age > 0:
                        break
                except (ValueError, TypeError):
                    continue
        
        # If no age found, try to calculate from date_of_birth
        if age is None or age <= 0:
            for dob_field in ["date_of_birth", "dob", "birth_date"]:
                if dob_field in profile_data and profile_data[dob_field] is not None:
                    calculated = self._calculate_age(profile_data[dob_field])
                    if calculated is not None and calculated > 0:
                        age = calculated
                        break

        status_value = await self._patient_status_from_user_activity(user_doc)
        flagged_critical = bool(profile_data.get("flagged_critical", False))

        return PatientAdminItem(
            id=formatted_id,
            user_id=user["id"],
            name=user.get("full_name", ""),
            age=age,
            status=status_value,
            flagged_critical=flagged_critical,
            family_history=self._normalize_text_list(
                profile_data.get("family_history") or profile_data.get("medical_history")
            ),
            gender=profile_data.get("gender"),
            phone=profile_data.get("phone"),
            blood_group=profile_data.get("blood_group") or profile_data.get("bloodGroup"),
            allergies=self._normalize_text_list(profile_data.get("allergies")),
            chronic_diseases=self._normalize_text_list(
                profile_data.get("chronic_diseases") or profile_data.get("chronicDiseases")
            ),
            emergency_contact=profile_data.get("emergency_contact") or profile_data.get("emergencyContact"),
        )

    async def list_doctors(self) -> list[DoctorAdminItem]:
        cursor = self.db.users.find({"role": "doctor"}).sort("created_at", -1)
        items: list[DoctorAdminItem] = []
        async for user in cursor:
            items.append(await self._build_doctor_admin_item(user))
        return items

    async def patch_doctor(self, doctor_id: str, payload: DoctorAdminPatchRequest) -> DoctorAdminItem:
        object_id, doctor_user = await self._resolve_doctor_user(doctor_id)

        doctor_updates: dict = {"updated_at": utcnow()}
        user_updates: dict = {"updated_at": utcnow()}

        if payload.status is not None:
            if payload.status == DoctorManagementStatus.ACTIVE:
                user_updates.update({"is_active": True, "is_blocked": False})
            else:
                user_updates.update({"is_active": False, "is_blocked": True})

        if payload.verification is not None:
            doctor_updates["verification"] = payload.verification.value
            doctor_updates["is_verified"] = payload.verification == DoctorVerificationStatus.APPROVED

            # Approval/rejection must also drive status so the admin UI stays in sync
            if payload.verification == DoctorVerificationStatus.APPROVED:
                user_updates.update({"is_active": True, "is_blocked": False})
            elif payload.verification == DoctorVerificationStatus.REJECTED:
                user_updates.update({"is_active": False, "is_blocked": True})

        doctor_updates["admin_role"] = DoctorManagementRole.DOCTOR.value

        if len(user_updates) > 1:
            await self.db.users.update_one({"_id": object_id}, {"$set": user_updates})

        if len(doctor_updates) > 1:
            await self.db.doctors.update_one(
                {"user_id": object_id},
                {"$set": doctor_updates, "$setOnInsert": {"created_at": utcnow()}},
                upsert=True,
            )

        updated_user = await self.db.users.find_one({"_id": object_id, "role": "doctor"})
        return await self._build_doctor_admin_item(updated_user)

    async def list_patients(self) -> list[PatientAdminItem]:
        cursor = self.db.users.find({"role": "patient"}).sort("created_at", -1)
        items: list[PatientAdminItem] = []
        async for user in cursor:
            items.append(await self._build_patient_admin_item(user))
        return items

    async def patch_patient(self, patient_id: str, payload: PatientAdminPatchRequest) -> PatientAdminItem:
        # Decode formatted ID if necessary (PAT-XXXX-XXXX format)
        decoded_id = decode_formatted_id(patient_id, "PAT") or patient_id
        
        # Try to find by patient profile ID first, then by user ID
        patient_profile = None
        patient_user = None
        
        try:
            profile_oid = ObjectId(decoded_id)
            patient_profile = await self.db.patients.find_one({"_id": profile_oid})
            if patient_profile:
                patient_user = await self.db.users.find_one({"_id": patient_profile["user_id"], "role": "patient"})
        except (InvalidId, TypeError):
            pass
        
        # If not found by profile ID, try by user ID
        if not patient_user:
            try:
                user_oid = ObjectId(decoded_id)
                patient_user = await self.db.users.find_one({"_id": user_oid, "role": "patient"})
                if patient_user:
                    patient_profile = await self.db.patients.find_one({"user_id": user_oid})
            except (InvalidId, TypeError):
                pass
        
        if not patient_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        user_updates: dict = {"updated_at": utcnow()}
        patient_updates: dict = {"updated_at": utcnow()}

        if payload.name is not None:
            user_updates["full_name"] = payload.name

        if payload.age is not None:
            patient_updates["age"] = payload.age

        if payload.status is not None:
            if payload.status == PatientManagementStatus.ACTIVE:
                user_updates.update({"is_active": True, "is_blocked": False})
            else:
                user_updates.update({"is_active": False, "is_blocked": True})

        if payload.flagged_critical is not None:
            patient_updates["flagged_critical"] = payload.flagged_critical

        if payload.family_history is not None:
            patient_updates["family_history"] = payload.family_history
            patient_updates["medical_history"] = payload.family_history

        if len(user_updates) > 1:
            await self.db.users.update_one({"_id": patient_user["_id"]}, {"$set": user_updates})

        if len(patient_updates) > 1:
            await self.db.patients.update_one(
                {"user_id": patient_user["_id"]},
                {"$set": patient_updates, "$setOnInsert": {"created_at": utcnow()}},
                upsert=True,
            )

        updated_user = await self.db.users.find_one({"_id": patient_user["_id"], "role": "patient"})
        return await self._build_patient_admin_item(updated_user)

    async def delete_patient(self, patient_id: str) -> bool:
        # Decode formatted ID if necessary (PAT-XXXX-XXXX format)
        decoded_id = decode_formatted_id(patient_id, "PAT") or patient_id
        
        # Try to find by patient profile ID first, then by user ID
        patient_profile = None
        patient_user = None
        
        try:
            profile_oid = ObjectId(decoded_id)
            patient_profile = await self.db.patients.find_one({"_id": profile_oid})
            if patient_profile:
                patient_user = await self.db.users.find_one({"_id": patient_profile["user_id"], "role": "patient"})
        except (InvalidId, TypeError):
            pass
        
        # If not found by profile ID, try by user ID
        if not patient_user:
            try:
                user_oid = ObjectId(decoded_id)
                patient_user = await self.db.users.find_one({"_id": user_oid, "role": "patient"})
            except (InvalidId, TypeError):
                pass
        
        if not patient_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        await self.db.patients.delete_many({"user_id": patient_user["_id"]})
        result = await self.db.users.delete_one({"_id": patient_user["_id"], "role": "patient"})
        return bool(result.deleted_count)

    async def list_reports(self, limit: int = 100) -> list[dict]:
        cursor = self.db.test_reports.find({}).sort("created_at", -1).limit(limit)
        items = []
        async for report in cursor:
            items.append(to_str_id(report))
        return items
