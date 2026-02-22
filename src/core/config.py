"""
ForgeMind AI Suite — Centralized Configuration
All settings loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "ForgeMind-AI"
    app_version: str = "2.0.0"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_reload: bool = True

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "forgemind"
    postgres_user: str = "forgemind"
    postgres_password: str = "changeme"
    database_url: Optional[str] = None

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_database_url(cls, v: Optional[str], info) -> str:
        if v:
            return v
        data = info.data
        return (
            f"postgresql+asyncpg://{data.get('postgres_user', 'forgemind')}:"
            f"{data.get('postgres_password', 'changeme')}@"
            f"{data.get('postgres_host', 'localhost')}:"
            f"{data.get('postgres_port', 5432)}/"
            f"{data.get('postgres_db', 'forgemind')}"
        )

    # --- TimescaleDB ---
    timescale_host: str = "localhost"
    timescale_port: int = 5433
    timescale_db: str = "forgemind_ts"
    timescale_user: str = "forgemind"
    timescale_password: str = "changeme"
    timescale_url: Optional[str] = None

    @field_validator("timescale_url", mode="before")
    @classmethod
    def assemble_timescale_url(cls, v: Optional[str], info) -> str:
        if v:
            return v
        data = info.data
        return (
            f"postgresql+asyncpg://{data.get('timescale_user', 'forgemind')}:"
            f"{data.get('timescale_password', 'changeme')}@"
            f"{data.get('timescale_host', 'localhost')}:"
            f"{data.get('timescale_port', 5433)}/"
            f"{data.get('timescale_db', 'forgemind_ts')}"
        )

    # --- Apache Kafka ---
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "forgemind-consumers"
    kafka_auto_offset_reset: str = "latest"
    kafka_enable_auto_commit: bool = True

    # --- MQTT ---
    mqtt_broker_host: str = "localhost"
    mqtt_broker_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_client_id: str = "forgemind-mqtt"

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_url: Optional[str] = None

    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_url(cls, v: Optional[str], info) -> str:
        if v:
            return v
        data = info.data
        password = data.get("redis_password", "")
        host = data.get("redis_host", "localhost")
        port = data.get("redis_port", 6379)
        db = data.get("redis_db", 0)
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

    # --- OPC-UA ---
    opcua_server_url: str = "opc.tcp://localhost:4840"
    opcua_namespace_uri: str = "urn:forgemind:opcua"
    opcua_security_policy: str = "Basic256Sha256"

    # --- JWT / Auth ---
    jwt_secret_key: str = "change-me-use-openssl-rand-hex-64"
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    oauth2_provider_url: str = ""

    # --- Observability ---
    otel_service_name: str = "forgemind-ai"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_port: int = 9090
    grafana_port: int = 3000

    # --- External APIs ---
    sap_erp_endpoint: str = ""
    sap_erp_api_key: str = ""
    dnb_api_key: str = ""
    coface_api_key: str = ""
    weather_api_key: str = ""

    # --- ML Ops ---
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_registry_path: str = "./models"

    # --- Vault ---
    vault_addr: str = "http://localhost:8200"
    vault_token: str = ""

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == Environment.DEVELOPMENT

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent


@lru_cache()
def get_settings() -> AppSettings:
    """Return cached singleton application settings."""
    return AppSettings()
