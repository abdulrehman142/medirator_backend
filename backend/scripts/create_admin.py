import asyncio
import sys

from app.core.security import hash_password
from app.db.mongo import close_mongo, init_mongo
from app.utils.time import utcnow


async def create_admin() -> None:
    db = init_mongo()

    print("\n=== Create Admin User ===\n")

    email = input("Enter admin email: ").strip().lower()
    if not email or len(email) < 3:
        print("Invalid email")
        return

    existing = await db.users.find_one({"email": email})
    if existing:
        print(f"User with email {email} already exists")
        await close_mongo()
        return

    full_name = input("Enter full name: ").strip()
    if not full_name or len(full_name) < 2:
        print("Full name too short (min 2 chars)")
        return

    password = input("Enter password (min 8 chars): ").strip()
    if not password or len(password) < 8:
        print("Password too short (min 8 chars)")
        return

    res = await db.users.insert_one(
        {
            "email": email,
            "full_name": full_name,
            "role": "admin",
            "hashed_password": hash_password(password),
            "is_active": True,
            "is_blocked": False,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
    )

    print(f"\nAdmin user created successfully!")
    print(f"ID: {res.inserted_id}")
    print(f"Email: {email}")
    print(f"Name: {full_name}")

    await close_mongo()


if __name__ == "__main__":
    asyncio.run(create_admin())
