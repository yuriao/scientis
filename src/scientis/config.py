"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────
    app_name: str = "scientis"
    debug: bool = False
    api_prefix: str = "/v1"

    # ── Neo4j ────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "scientis-dev"

    # ── PostgreSQL ───────────────────────────────────
    database_url: str = "postgresql+asyncpg://scientis:scientis-dev@localhost:5432/scientis"

    # ── Redis ────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Object Storage (S3/MinIO) ────────────────────
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "scientis-papers"
    s3_region: str = "us-east-1"

    # ── LLM Providers ────────────────────────────────
    openai_api_key: str = ""
    gemini_api_key: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

    # ── Observability (optional) ─────────────────────
    langsmith_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
