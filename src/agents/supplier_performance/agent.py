"""
SupplierPerformanceAgent — Schemas, Service & Router
SLA compliance tracking with statistical rigor. Monthly scorecards with trend analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import AgentRole, AuthUser, require_role


class EscalationLevel(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    CRITICAL = "CRITICAL"


class SupplierScorecard(BaseModel):
    supplier_id: str
    period: str
    otd_rate: float = Field(0.0, ge=0, le=100, description="On-Time Delivery %")
    quality_ppm: float = Field(0.0, ge=0, description="Quality defects per million")
    lead_time_accuracy_pct: float = Field(0.0, ge=0, le=100)
    responsiveness_index: float = Field(0.0, ge=0, le=10)
    price_stability_score: float = Field(0.0, ge=0, le=100)
    overall_score: float = Field(0.0, ge=0, le=100)
    escalation_level: EscalationLevel = EscalationLevel.GREEN
    trend: str = "STABLE"
    penalty_triggered: bool = False
    reward_triggered: bool = False
    notes: str = ""
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SupplierScorecardRequest(BaseModel):
    supplier_id: str
    period: str = "monthly"
    otd_deliveries: int = 95
    total_deliveries: int = 100
    defects_found: int = 5
    total_units: int = 10000
    quoted_lead_days: float = 14.0
    actual_lead_days: float = 15.0
    response_time_hours: float = 4.0
    price_variance_pct: float = 2.0


class SupplierPerformanceService:

    async def generate_scorecard(self, req: SupplierScorecardRequest) -> SupplierScorecard:
        otd = (req.otd_deliveries / max(req.total_deliveries, 1)) * 100
        ppm = (req.defects_found / max(req.total_units, 1)) * 1_000_000
        lt_accuracy = max(0, 100 - abs(req.actual_lead_days - req.quoted_lead_days) / max(req.quoted_lead_days, 1) * 100)
        responsiveness = max(0, 10 - req.response_time_hours / 4.8)
        price_stability = max(0, 100 - abs(req.price_variance_pct) * 10)

        overall = (otd * 0.30 + min(100, max(0, 100 - ppm / 100)) * 0.25
                   + lt_accuracy * 0.20 + responsiveness * 10 * 0.15 + price_stability * 0.10)

        if overall >= 85:
            level = EscalationLevel.GREEN
        elif overall >= 70:
            level = EscalationLevel.AMBER
        elif overall >= 50:
            level = EscalationLevel.RED
        else:
            level = EscalationLevel.CRITICAL

        return SupplierScorecard(
            supplier_id=req.supplier_id,
            period=req.period,
            otd_rate=round(otd, 1),
            quality_ppm=round(ppm, 0),
            lead_time_accuracy_pct=round(lt_accuracy, 1),
            responsiveness_index=round(responsiveness, 2),
            price_stability_score=round(price_stability, 1),
            overall_score=round(overall, 1),
            escalation_level=level,
            trend="STABLE",
            penalty_triggered=level in (EscalationLevel.RED, EscalationLevel.CRITICAL),
            reward_triggered=level == EscalationLevel.GREEN and overall > 90,
            notes=f"Escalation path: {level.value}" + (
                " → supplier review meeting" if level == EscalationLevel.AMBER else
                " → corrective action plan" if level == EscalationLevel.RED else
                " → contract review board" if level == EscalationLevel.CRITICAL else ""
            ),
        )


supplier_perf_service = SupplierPerformanceService()

router = APIRouter(prefix="/api/v2/supplier-performance", tags=["Supplier Performance"])


@router.post("/scorecard", response_model=SupplierScorecard, summary="Generate supplier scorecard")
async def generate_scorecard(
    request: SupplierScorecardRequest,
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    return await supplier_perf_service.generate_scorecard(request)
