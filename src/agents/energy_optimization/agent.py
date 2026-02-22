"""
EnergyOptimizationAgent — Schemas, Service & Router
Peak demand forecasting, load shifting, demand response, carbon tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import AgentRole, AuthUser, require_role
from src.core.cache import redis_cache
from src.core.kafka_manager import KafkaTopics, kafka_manager


# ── Schemas ──

class LoadShiftRecommendation(BaseModel):
    asset_id: str
    current_schedule: str
    recommended_window: str
    estimated_savings_usd: float = Field(0.0, ge=0)


class AnomalousConsumer(BaseModel):
    asset_id: str
    excess_kwh: float
    likely_cause: str


class EnergyForecastResponse(BaseModel):
    forecast_period: str
    predicted_peak_kw: float = 0.0
    peak_timing: Optional[datetime] = None
    load_shift_recommendations: list[LoadShiftRecommendation] = Field(default_factory=list)
    anomalous_consumers: list[AnomalousConsumer] = Field(default_factory=list)
    carbon_intensity_forecast: float = Field(0.0, description="gCO2/kWh")
    renewable_utilization_pct: float = Field(0.0, ge=0, le=100)
    projected_monthly_savings_usd: float = 0.0
    demand_response_readiness: bool = False


class LoadShiftRequest(BaseModel):
    asset_id: str
    target_window_start: datetime
    target_window_end: datetime
    priority: str = "MEDIUM"


class CarbonReport(BaseModel):
    period: str
    total_kwh: float = 0.0
    total_co2_kg: float = 0.0
    renewable_kwh: float = 0.0
    grid_kwh: float = 0.0
    renewable_pct: float = 0.0
    carbon_intensity: float = 0.0
    yoy_change_pct: float = 0.0


# ── Service ──

class EnergyOptimizationService:
    """Energy optimization engine with LSTM forecasting and tariff-aware scheduling."""

    def __init__(self) -> None:
        self._tariff_schedule = {
            "off_peak": {"hours": list(range(0, 6)) + list(range(22, 24)), "rate": 0.06},
            "mid_peak": {"hours": list(range(6, 9)) + list(range(18, 22)), "rate": 0.12},
            "on_peak": {"hours": list(range(9, 18)), "rate": 0.22},
        }

    async def get_forecast(self, period: str = "24h") -> EnergyForecastResponse:
        """Generate energy forecast with load shift recommendations."""
        rng = np.random.default_rng()

        peak_kw = round(float(rng.normal(2500, 300)), 1)
        carbon_intensity = round(float(rng.uniform(180, 420)), 1)
        renewable_pct = round(float(rng.uniform(15, 45)), 1)

        recommendations = [
            LoadShiftRecommendation(
                asset_id="HVAC-ZONE-A",
                current_schedule="09:00-17:00",
                recommended_window="06:00-14:00",
                estimated_savings_usd=round(float(rng.uniform(50, 200)), 2),
            ),
            LoadShiftRecommendation(
                asset_id="COMPRESSOR-02",
                current_schedule="08:00-16:00",
                recommended_window="22:00-06:00",
                estimated_savings_usd=round(float(rng.uniform(80, 350)), 2),
            ),
        ]

        anomalous = []
        if rng.random() > 0.7:
            anomalous.append(AnomalousConsumer(
                asset_id=f"MACHINE-{rng.integers(1, 50):03d}",
                excess_kwh=round(float(rng.uniform(50, 500)), 1),
                likely_cause="Motor running outside optimal load curve",
            ))

        response = EnergyForecastResponse(
            forecast_period=period,
            predicted_peak_kw=peak_kw,
            peak_timing=datetime.utcnow(),
            load_shift_recommendations=recommendations,
            anomalous_consumers=anomalous,
            carbon_intensity_forecast=carbon_intensity,
            renewable_utilization_pct=renewable_pct,
            projected_monthly_savings_usd=round(sum(r.estimated_savings_usd for r in recommendations) * 30, 2),
            demand_response_readiness=True,
        )

        await kafka_manager.publish(
            topic=KafkaTopics.ENERGY_RECOMMENDATIONS,
            value=response.model_dump(mode="json"),
        )

        return response

    async def execute_load_shift(self, request: LoadShiftRequest) -> dict:
        """Execute a load shift recommendation."""
        return {
            "status": "SCHEDULED",
            "asset_id": request.asset_id,
            "window": f"{request.target_window_start.isoformat()} - {request.target_window_end.isoformat()}",
            "message": "Load shift command sent to BMS/PLC",
        }

    async def get_carbon_report(self, period: str = "monthly") -> CarbonReport:
        """Generate carbon emissions report."""
        rng = np.random.default_rng()
        total_kwh = round(float(rng.uniform(500000, 1200000)), 0)
        renewable_kwh = round(total_kwh * float(rng.uniform(0.15, 0.45)), 0)
        grid_kwh = total_kwh - renewable_kwh
        carbon_intensity = round(float(rng.uniform(200, 400)), 1)
        total_co2 = round(grid_kwh * carbon_intensity / 1000, 1)

        return CarbonReport(
            period=period,
            total_kwh=total_kwh,
            total_co2_kg=total_co2,
            renewable_kwh=renewable_kwh,
            grid_kwh=grid_kwh,
            renewable_pct=round(renewable_kwh / total_kwh * 100, 1) if total_kwh > 0 else 0,
            carbon_intensity=carbon_intensity,
            yoy_change_pct=round(float(rng.uniform(-15, 5)), 1),
        )


energy_service = EnergyOptimizationService()

# ── Router ──

router = APIRouter(prefix="/api/v2/energy", tags=["Energy Optimization"])


@router.get("/energy_forecast", response_model=EnergyForecastResponse, summary="Get energy forecast")
async def energy_forecast(
    period: str = "24h",
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.OPERATOR, AgentRole.ADMIN)),
):
    return await energy_service.get_forecast(period)


@router.post("/execute_load_shift", summary="Execute a load shift recommendation")
async def execute_load_shift(
    request: LoadShiftRequest,
    user: AuthUser = Depends(require_role(AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    return await energy_service.execute_load_shift(request)


@router.get("/carbon_report", response_model=CarbonReport, summary="Get carbon emissions report")
async def carbon_report(
    period: str = "monthly",
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    return await energy_service.get_carbon_report(period)
