"""
ProductionSchedulerAgent — Pydantic Schemas
Constraint satisfaction & optimization engine for manufacturing scheduling.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ──

class JobPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    STANDARD = "STANDARD"
    LOW = "LOW"


# ── Input Schemas ──

class JobOrder(BaseModel):
    """ERP job order for scheduling."""

    job_id: str
    product_sku: str
    quantity: int = Field(..., gt=0)
    due_date: datetime
    priority: JobPriority = JobPriority.STANDARD
    bom_id: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    estimated_runtime_hours: float = Field(1.0, gt=0)
    setup_time_min: int = Field(0, ge=0)
    locked: bool = False  # If true, never reschedule


class MachineCapability(BaseModel):
    """Machine registry entry."""

    machine_id: str
    capabilities: list[str] = Field(default_factory=list)
    setup_time_matrix: dict[str, int] = Field(default_factory=dict)
    oee_history: list[float] = Field(default_factory=list)
    available_hours_per_day: float = Field(16.0, gt=0)
    is_available: bool = True
    maintenance_windows: list[dict[str, datetime]] = Field(default_factory=list)


class LaborSchedule(BaseModel):
    """Shift labor information."""

    shift_id: str
    workers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    start_time: datetime
    end_time: datetime
    overtime_cap_hours: float = Field(4.0, ge=0)


class ScheduleRequest(BaseModel):
    """Full request to the scheduling optimizer."""

    jobs: list[JobOrder]
    machines: list[MachineCapability]
    labor: list[LaborSchedule] = Field(default_factory=list)
    horizon_days: int = Field(7, ge=1, le=30)
    locked_jobs: list[str] = Field(default_factory=list)
    maintenance_windows: list[dict] = Field(default_factory=list)


# ── Output Schemas ──

class ScheduledJob(BaseModel):
    """Single job assignment in the optimized schedule."""

    job_id: str
    machine_id: str
    shift_id: str = ""
    start_time: datetime
    end_time: datetime
    operator_id: str = ""
    setup_time_min: int = 0
    buffer_time_min: int = 15


class ScheduleKPIs(BaseModel):
    """KPI summary for the generated schedule."""

    projected_oee: float = Field(0.0, ge=0.0, le=1.0)
    bottleneck_machines: list[str] = Field(default_factory=list)
    on_time_delivery_rate: float = Field(0.0, ge=0.0, le=1.0)
    total_overtime_hours: float = Field(0.0, ge=0.0)
    changeover_cost_index: float = Field(0.0, ge=0.0)


class InfeasibleJob(BaseModel):
    """Job that could not be scheduled."""

    job_id: str
    reason: str


class ScheduleResponse(BaseModel):
    """Full optimization response."""

    schedule_id: UUID = Field(default_factory=uuid4)
    generated_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    horizon_days: int = 7
    schedule: list[ScheduledJob] = Field(default_factory=list)
    kpis: ScheduleKPIs = Field(default_factory=ScheduleKPIs)
    infeasible_jobs: list[InfeasibleJob] = Field(default_factory=list)
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    solver_runtime_ms: int = 0
    optimality_gap: Optional[float] = None


class BottleneckAnalysis(BaseModel):
    """Bottleneck analysis result."""

    machine_id: str
    utilization_pct: float
    queue_depth: int
    average_wait_hours: float
    is_bottleneck: bool
    recommendation: str
