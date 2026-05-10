import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Medirator API"
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = os.getenv("MONGO_DB_NAME", "medirator_db")

    redis_url: str = "redis://localhost:6379/0"

    jwt_secret_key: str = Field("change-this-super-secret-key-please-override", min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    password_reset_token_expire_minutes: int = 30

    allowed_origins: str = "http://localhost:5173,http://localhost:3000,https://medirator.netlify.app"
    rate_limit_per_minute: int = 120
    failed_login_limit: int = 5
    block_duration_minutes: int = 30
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("gemini_api_key", "GEMINI_API_KEY"),
    )
    gemini_enabled: bool = Field(default=True, validation_alias=AliasChoices("gemini_enabled", "GEMINI_ENABLED"))
    xrayas_model_path: str = "models/chest_xray_model_full.pkl"
    huggingface_hub_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("huggingface_hub_token", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
    )

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    @field_validator("gemini_api_key", "huggingface_hub_token", mode="before")
    @classmethod
    def _strip_optional_secrets(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return v

    @property
    def allowed_origins_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
        netlify_origin = "https://medirator.netlify.app"
        if netlify_origin not in origins:
            origins.append(netlify_origin)
        return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
