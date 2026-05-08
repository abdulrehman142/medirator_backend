from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.errors import InvalidId
from datetime import date

from app.schemas.user import DoctorDirectoryItem, DoctorProfile, PatientDirectoryItem, PatientProfile, DoctorPatientProfile, UserCreate, UserPublic, UserUpdate
from app.utils.bson_utils import to_str_id
from app.utils.time import utcnow
from app.utils.id_formatter import format_patient_id, format_doctor_id


class UserService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def get_user_by_id(self, user_id: str) -> dict | None:
        try:
            object_id = ObjectId(user_id)
        except (InvalidId, TypeError):
            return None
        user = await self.db.users.find_one({"_id": object_id})
        return to_str_id(user) if user else None

    async def create_user(self, user_in: UserCreate, hashed_password: str) -> UserPublic:
        payload = {
            "email": user_in.email.lower(),
            "full_name": user_in.full_name,
            "role": user_in.role.value,
            "hashed_password": hashed_password,
            "is_active": True,
            "is_blocked": False,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        res = await self.db.users.insert_one(payload)

        if user_in.role.value == "patient":
            await self.db.patients.insert_one({"user_id": res.inserted_id, "created_at": utcnow(), "updated_at": utcnow()})
        if user_in.role.value == "doctor":
            await self.db.doctors.insert_one({"user_id": res.inserted_id, "created_at": utcnow(), "updated_at": utcnow()})

        return await self.get_user_public_by_id(str(res.inserted_id))

    async def get_user_by_email(self, email: str) -> dict | None:
        user = await self.db.users.find_one({"email": email.lower()})
        return to_str_id(user) if user else None

    async def get_user_public_by_id(self, user_id: str) -> UserPublic | None:
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        return UserPublic(
            id=user["id"],
            email=user["email"],
            full_name=user["full_name"],
            role=user["role"],
            is_active=user["is_active"],
        )

    async def list_patients(self) -> list[PatientDirectoryItem]:
        items: list[PatientDirectoryItem] = []
        cursor = self.db.users.find({"role": "patient", "is_active": True, "is_blocked": False}).sort("created_at", -1)
        async for user in cursor:
            transformed = to_str_id(user)
            profile = await self.get_patient_profile(transformed["id"])
            items.append(
                PatientDirectoryItem(
                    id=transformed["id"],
                    display_id=format_patient_id(transformed["id"]),
                    name=transformed.get("full_name", ""),
                    email=transformed["email"],
                    status="active" if transformed.get("is_active") and not transformed.get("is_blocked") else "inactive",
                    date_of_birth=profile.date_of_birth if profile else None,
                    gender=profile.gender if profile else None,
                    phone=profile.phone if profile else None,
                    emergency_contact=profile.emergency_contact if profile else None,
                    blood_group=profile.blood_group if profile else None,
                    allergies=profile.allergies if profile else None,
                    chronic_diseases=profile.chronic_diseases if profile else None,
                    family_history=profile.family_history if profile else None,
                )
            )
        return items

    async def list_doctor_directory(self) -> list[DoctorDirectoryItem]:
        items: list[DoctorDirectoryItem] = []
        cursor = self.db.users.find({"role": "doctor", "is_active": True, "is_blocked": False}).sort("created_at", -1)
        async for user in cursor:
            transformed = to_str_id(user)
            profile = await self.get_doctor_profile(transformed["id"])
            items.append(
                DoctorDirectoryItem(
                    id=transformed["id"],
                    display_id=format_doctor_id(transformed["id"]),
                    name=transformed.get("full_name", ""),
                    specialization=profile.specialization if profile else None,
                    status="active" if transformed.get("is_active") and not transformed.get("is_blocked") else "inactive",
                    is_verified=profile.is_verified if profile else False,
                )
            )
        return items

    async def list_doctors(self, only_active: bool = True, only_verified: bool = True) -> list[dict]:
        query = {"role": "doctor"}
        if only_active:
            query.update({"is_active": True, "is_blocked": False})
        items: list[dict] = []
        cursor = self.db.users.find(query).sort("created_at", -1)
        async for user in cursor:
            user = to_str_id(user)
            profile = await self.get_doctor_profile(user["id"])
            if only_verified and (not profile or not (profile.license_number or profile.is_verified)):
                continue
            items.append({**user, "profile": profile.model_dump() if profile else None})
        return items

    async def update_user(self, user_id: str, user_in: UserUpdate) -> UserPublic | None:
        payload = {k: v for k, v in user_in.model_dump().items() if v is not None}
        if not payload:
            return await self.get_user_public_by_id(user_id)
        payload["updated_at"] = utcnow()
        await self.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": payload})
        return await self.get_user_public_by_id(user_id)

    async def set_user_blocked(self, user_id: str, blocked: bool) -> None:
        await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_blocked": blocked, "updated_at": utcnow()}},
        )

    async def get_patient_profile(self, user_id: str) -> PatientProfile | None:
        doc = await self.db.patients.find_one({"user_id": ObjectId(user_id)})
        if not doc:
            return None
        user = await self.get_user_by_id(user_id)
        payload = {k: v for k, v in doc.items() if k not in {"_id", "user_id", "created_at", "updated_at"}}
        if "family_history" not in payload and "medical_history" in payload:
            payload["family_history"] = payload.get("medical_history")
        return PatientProfile(user_id=user_id, full_name=user["full_name"] if user else None, **payload)

    async def upsert_patient_profile(self, user_id: str, profile: PatientProfile) -> PatientProfile:
        payload = profile.model_dump(exclude={"user_id"}, exclude_none=True)
        if "family_history" in payload:
            payload["medical_history"] = payload["family_history"]
        full_name = payload.pop("full_name", None)
        if full_name:
            await self.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"full_name": full_name, "updated_at": utcnow()}})
        payload["updated_at"] = utcnow()
        await self.db.patients.update_one(
            {"user_id": ObjectId(user_id)},
            {"$set": payload, "$setOnInsert": {"created_at": utcnow()}},
            upsert=True,
        )
        return await self.get_patient_profile(user_id)

    async def get_doctor_profile(self, user_id: str) -> DoctorProfile | None:
        doc = await self.db.doctors.find_one({"user_id": ObjectId(user_id)})
        if not doc:
            return None
        user = await self.get_user_by_id(user_id)
        payload = {k: v for k, v in doc.items() if k not in {"_id", "user_id", "created_at", "updated_at"}}
        return DoctorProfile(user_id=user_id, full_name=user["full_name"] if user else None, **payload)

    async def upsert_doctor_profile(self, user_id: str, profile: DoctorProfile) -> DoctorProfile:
        payload = profile.model_dump(exclude={"user_id"}, exclude_none=True)
        full_name = payload.pop("full_name", None)
        if full_name:
            await self.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"full_name": full_name, "updated_at": utcnow()}})
        payload["updated_at"] = utcnow()
        await self.db.doctors.update_one(
            {"user_id": ObjectId(user_id)},
            {"$set": payload, "$setOnInsert": {"created_at": utcnow()}},
            upsert=True,
        )
        return await self.get_doctor_profile(user_id)

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
        """Normalize list/string medical fields into a UI-friendly string."""
        if value is None:
            return None
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(cleaned) if cleaned else None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return str(value)

    async def get_doctor_patient_profile(self, patient_user_id: str) -> DoctorPatientProfile | None:
        """
        Get complete patient profile as seen by doctors.
        Returns all available fields with normalized names.
        """
        try:
            user_oid = ObjectId(patient_user_id)
        except (InvalidId, TypeError):
            return None
        
        user = await self.db.users.find_one({"_id": user_oid, "role": "patient"})
        if not user:
            return None
        
        patient_doc = await self.db.patients.find_one({"user_id": user_oid})
        profile_data = to_str_id(patient_doc) if patient_doc else {}
        
        # Extract age from various possible field names
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
        
        # Format patient ID consistently
        patient_id = profile_data.get("id") or str(user_oid)
        formatted_id = format_patient_id(patient_id)
        
        return DoctorPatientProfile(
            id=formatted_id,
            name=user.get("full_name", ""),
            age=age,
            gender=profile_data.get("gender"),
            contact=profile_data.get("phone") or profile_data.get("contact"),
            blood_group=profile_data.get("blood_group") or profile_data.get("bloodGroup"),
            allergies=self._normalize_text_list(profile_data.get("allergies")),
            chronic_diseases=self._normalize_text_list(
                profile_data.get("chronic_diseases") or profile_data.get("chronicDiseases")
            ),
            emergency_contact=profile_data.get("emergency_contact") or profile_data.get("emergencyContact"),
            family_history=self._normalize_text_list(
                profile_data.get("family_history") or profile_data.get("medical_history") or profile_data.get("diagnosis")
            ),
            family_tree=profile_data.get("family_tree") or [],
            doctor_notes=profile_data.get("doctor_notes", []),
            uploaded_documents=profile_data.get("uploaded_documents", []),
        )
