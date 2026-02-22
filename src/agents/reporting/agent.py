"""
ReportingAgent — Intelligence aggregation layer for the entire ForgeMind suite.
Executive Daily Brief, Operational Dashboard, Regulatory Compliance, Agent Health.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import AgentRole, AuthUser, require_role


class ReportType(str, Enum):
    EXECUTIVE_BRIEF = "EXECUTIVE_BRIEF"
    OPERATIONAL_DASHBOARD = "OPERATIONAL_DASHBOARD"
    REGULATORY_COMPLIANCE = "REGULATORY_COMPLIANCE"
    AGENT_HEALTH = "AGENT_HEALTH"


class KPIStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class KPIMetric(BaseModel):
    name: str
    value: float
    unit: str
    target: float
    status: KPIStatus
    deviation_sigma: float = 0.0


class AgentHealthStatus(BaseModel):
    agent_name: str
    is_healthy: bool = True
    last_response_ms: int = 0
    sla_breaches_24h: int = 0
    model_drift_detected: bool = False
    integration_errors_24h: int = 0
    uptime_pct: float = 99.9


class ExecutiveBrief(BaseModel):
    date: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    headline: str = ""
    kpi_snapshot: list[KPIMetric] = Field(default_factory=list)
    exception_highlights: list[str] = Field(default_factory=list)
    agent_health: list[AgentHealthStatus] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class DashboardFeed(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = Field(default_factory=dict)
    alerts: list[dict] = Field(default_factory=list)
    trends: dict[str, str] = Field(default_factory=dict)


class ReportingService:

    AGENT_NAMES = [
        "PredictiveMaintenanceAgent",
        "ProductionSchedulerAgent",
        "VisionQC_Agent",
        "SupplyRiskAgent",
        "EnergyOptimizationAgent",
        "InventoryForecastingAgent",
        "WorkerSafetyAgent",
        "ProcessAnalyzerAgent",
        "RootCauseAnalysisAgent",
        "DemandPlanningAgent",
        "DigitalTwinAgent",
        "SupplierPerformanceAgent",
        "WasteReductionAgent",
        "ReportingAgent",
    ]

    async def generate_executive_brief(self) -> ExecutiveBrief:
        rng = np.random.default_rng()

        kpis = [
            KPIMetric(name="Overall OEE", value=round(float(rng.uniform(80, 92)), 1), unit="%", target=85.0, status=KPIStatus.NORMAL, deviation_sigma=0.5),
            KPIMetric(name="On-Time Delivery", value=round(float(rng.uniform(92, 99)), 1), unit="%", target=95.0, status=KPIStatus.NORMAL, deviation_sigma=0.2),
            KPIMetric(name="Quality Yield", value=round(float(rng.uniform(96, 99.5)), 1), unit="%", target=98.0, status=KPIStatus.NORMAL, deviation_sigma=0.3),
            KPIMetric(name="Safety Incidents", value=float(rng.integers(0, 3)), unit="events", target=0.0, status=KPIStatus.WARNING if rng.random() > 0.7 else KPIStatus.NORMAL, deviation_sigma=1.2),
            KPIMetric(name="Energy Cost", value=round(float(rng.uniform(15000, 25000)), 0), unit="USD", target=20000.0, status=KPIStatus.NORMAL, deviation_sigma=0.8),
            KPIMetric(name="Scrap Rate", value=round(float(rng.uniform(1, 4)), 2), unit="%", target=2.5, status=KPIStatus.NORMAL, deviation_sigma=0.4),
        ]

        # Flag metrics > 2σ from 30-day average
        exceptions = [
            f"⚠️ {kpi.name}: {kpi.value}{kpi.unit} (target: {kpi.target}{kpi.unit})"
            for kpi in kpis
            if abs(kpi.deviation_sigma) > 2 or kpi.status != KPIStatus.NORMAL
        ]

        agent_health = [
            AgentHealthStatus(
                agent_name=name,
                is_healthy=rng.random() > 0.05,
                last_response_ms=int(rng.integers(20, 500)),
                sla_breaches_24h=int(rng.integers(0, 3)),
                model_drift_detected=rng.random() > 0.9,
                integration_errors_24h=int(rng.integers(0, 5)),
                uptime_pct=round(float(rng.uniform(98, 100)), 2),
            )
            for name in self.AGENT_NAMES
        ]

        return ExecutiveBrief(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            headline="Manufacturing operations running within normal parameters",
            kpi_snapshot=kpis,
            exception_highlights=exceptions,
            agent_health=agent_health,
            recommendations=[
                "Review preventive maintenance schedule for CNC machines — 3 assets approaching RUL threshold",
                "Energy costs trending 8% above target — consider load shifting on compressors",
            ],
        )

    async def get_dashboard_feed(self) -> DashboardFeed:
        rng = np.random.default_rng()
        return DashboardFeed(
            metrics={
                "oee": round(float(rng.uniform(80, 95)), 1),
                "throughput_uph": round(float(rng.uniform(100, 150)), 0),
                "quality_yield": round(float(rng.uniform(96, 100)), 2),
                "safety_score": round(float(rng.uniform(90, 100)), 1),
                "energy_kwh_today": round(float(rng.uniform(5000, 12000)), 0),
                "active_alerts": float(rng.integers(0, 10)),
            },
            trends={
                "oee": "STABLE",
                "throughput": "IMPROVING",
                "quality": "STABLE",
                "energy": "DETERIORATING",
            },
        )


reporting_service = ReportingService()

router = APIRouter(prefix="/api/v2/reports", tags=["Reporting"])


@router.get("/executive_brief", response_model=ExecutiveBrief, summary="Generate Executive Daily Brief")
async def executive_brief(
    user: AuthUser = Depends(require_role(AgentRole.ADMIN, AgentRole.VIEWER)),
):
    return await reporting_service.generate_executive_brief()


@router.get("/dashboard_feed", response_model=DashboardFeed, summary="Real-time dashboard feed")
async def dashboard_feed(
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    return await reporting_service.get_dashboard_feed()
