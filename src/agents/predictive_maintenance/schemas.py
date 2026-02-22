"""
PredictiveMaintenanceAgent — Pydantic Schemas
Strict contracts for telemetry input, prediction output, and maintenance actions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ──

class ActionType(str, Enum):
    INSPECT = "INSPECT"
    LUBRICATE = "LUBRICATE"
    REPLACE_PART = "REPLACE_PART"
    SHUTDOWN_IMMEDIATE = "SHUTDOWN_IMMEDIATE"
    MONITOR = "MONITOR"


class UrgencyLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class CriticalityTier(str, Enum):
    TIER_1 = "TIER_1"  # Mission-critical
    TIER_2 = "TIER_2"  # Important
    TIER_3 = "TIER_3"  # Standard
    TIER_4 = "TIER_4"  # Non-critical


# ── Input Schemas ──

class TelemetryReading(BaseModel):
    """Single telemetry reading from a machine sensor array."""

    machine_id: str = Field(..., description="Unique machine identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    temperature_C: float = Field(..., ge=-50, le=500, description="Temperature in Celsius")
    vibration_ms2: float = Field(..., ge=0, le=1000, description="Vibration in m/s²")
    pressure_bar: float = Field(..., ge=0, le=500, description="Pressure in bar")
    rpm: float = Field(..., ge=0, le=50000, description="Rotations per minute")
    current_A: float = Field(..., ge=0, le=5000, description="Current in Amperes")
    acoustic_dB: float = Field(0.0, ge=0, le=200, description="Acoustic level in dB")
    oil_viscosity_cSt: float = Field(0.0, ge=0, le=1000, description="Oil viscosity in cSt")

    @field_validator("machine_id")
    @classmethod
    def validate_machine_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("machine_id cannot be empty")
        return v.strip()


class MachineMetadata(BaseModel):
    """Machine asset metadata from the registry."""

    machine_id: str
    asset_class: str = ""
    install_date: Optional[datetime] = None
    oem_thresholds: dict[str, float] = Field(default_factory=dict)
    criticality_tier: CriticalityTier = CriticalityTier.TIER_3
    location: str = ""
    manufacturer: str = ""
    model_number: str = ""


class MaintenanceLog(BaseModel):
    """Historical maintenance record."""

    log_id: UUID = Field(default_factory=uuid4)
    machine_id: str
    timestamp: datetime
    action_type: ActionType
    description: str = ""
    technician_id: str = ""
    parts_used: list[str] = Field(default_factory=list)
    downtime_hours: float = 0.0
    cost: float = 0.0


# ── Output Schemas ──

class AnomalySignature(BaseModel):
    """Individual anomaly detection result per sensor."""

    sensor: str
    deviation_sigma: float = Field(..., description="Deviation from baseline in σ")
    pattern: str = Field("", description="Detected anomaly pattern type")


class RecommendedAction(BaseModel):
    """Actionable maintenance recommendation."""

    type: ActionType
    urgency: UrgencyLevel
    estimated_downtime_hours: float = Field(..., ge=0)
    parts_required: list[str] = Field(default_factory=list)
    work_order_trigger: bool = False


class PredictionResponse(BaseModel):
    """Full prediction output for a machine."""

    machine_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    failure_probability_72h: float = Field(..., ge=0.0, le=1.0)
    failure_probability_7d: float = Field(..., ge=0.0, le=1.0)
    rul_estimate_hours: int = Field(..., ge=0, description="Remaining Useful Life in hours")
    rul_confidence_interval: tuple[int, int] = Field(
        ..., description="90% confidence interval for RUL"
    )
    anomaly_signatures: list[AnomalySignature] = Field(default_factory=list)
    root_cause_hypothesis: str = ""
    recommended_action: RecommendedAction
    model_confidence: float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("rul_confidence_interval")
    @classmethod
    def validate_ci(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("Lower bound of CI must be <= upper bound")
        return v


class HealthDashboardResponse(BaseModel):
    """Aggregated health dashboard data."""

    machines: list[dict] = Field(default_factory=list)
    overall_health_score: float = 0.0
    critical_alerts: int = 0
    pending_work_orders: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())


class RULResponse(BaseModel):
    """Remaining Useful Life response for a single machine."""

    machine_id: str
    rul_estimate_hours: int
    rul_confidence_interval: tuple[int, int]
    model_confidence: float
    last_updated: datetime
