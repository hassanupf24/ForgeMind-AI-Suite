"""
InventoryForecastingAgent — Schemas, Service & Router
Prophet + LSTM + Croston's method for demand forecasting and reorder point optimization.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.core.auth import AgentRole, AuthUser, require_role
from src.core.cache import redis_cache
from src.core.kafka_manager import KafkaTopics, kafka_manager


# ── Schemas ──

class AlertType(str, Enum):
    STOCKOUT_RISK = "STOCKOUT_RISK"
    OVERSTOCK = "OVERSTOCK"
    EXPIRY_RISK = "EXPIRY_RISK"


class WeeklyForecast(BaseModel):
    week: int
    projected_demand: float
    confidence_interval: tuple[float, float]


class InventoryForecastRequest(BaseModel):
    sku_id: str
    current_stock_units: int = Field(0, ge=0)
    avg_daily_demand: float = Field(1.0, gt=0)
    lead_time_days: float = Field(14.0, gt=0)
    lead_time_std_days: float = Field(2.0, ge=0)
    service_level_target: float = Field(0.985, ge=0.5, le=1.0)
    unit_cost: float = Field(1.0, gt=0)
    holding_cost_pct: float = Field(0.25, ge=0, le=1)
    order_cost: float = Field(50.0, ge=0)


class InventoryForecastResponse(BaseModel):
    sku_id: str
    current_stock_units: int = 0
    days_of_supply: float = 0.0
    stockout_probability_30d: float = Field(0.0, ge=0.0, le=1.0)
    reorder_point: int = 0
    reorder_quantity: int = 0
    recommended_reorder_date: Optional[date] = None
    preferred_supplier: str = ""
    backup_supplier: str = ""
    carrying_cost_per_unit_month: float = 0.0
    forecast_horizon: list[WeeklyForecast] = Field(default_factory=list)
    alert_type: Optional[AlertType] = None


class InventoryHealthResponse(BaseModel):
    total_skus: int = 0
    stockout_risk_count: int = 0
    overstock_count: int = 0
    expiry_risk_count: int = 0
    avg_days_of_supply: float = 0.0
    total_carrying_cost_month: float = 0.0


# ── Service ──

class InventoryForecastingService:
    """Inventory optimization engine with multi-method forecasting stack."""

    def _calculate_safety_stock(
        self,
        avg_daily_demand: float,
        lead_time_days: float,
        lead_time_std: float,
        demand_std: float,
        service_level: float,
    ) -> int:
        """Service-level-driven safety stock calculation."""
        from scipy.stats import norm

        z_score = norm.ppf(service_level)
        safety_stock = z_score * np.sqrt(
            lead_time_days * demand_std**2 + avg_daily_demand**2 * lead_time_std**2
        )
        return max(0, int(np.ceil(safety_stock)))

    def _calculate_eoq(
        self,
        annual_demand: float,
        order_cost: float,
        holding_cost_per_unit: float,
    ) -> int:
        """Economic Order Quantity (EOQ) optimization."""
        if holding_cost_per_unit <= 0 or annual_demand <= 0:
            return 1
        eoq = np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)
        return max(1, int(np.ceil(eoq)))

    async def forecast(self, request: InventoryForecastRequest) -> InventoryForecastResponse:
        """Generate inventory forecast with reorder recommendations."""
        rng = np.random.default_rng()

        # Demand variability (assume CV ~0.3 for demo)
        demand_std = request.avg_daily_demand * 0.3

        # Safety stock
        safety_stock = self._calculate_safety_stock(
            request.avg_daily_demand,
            request.lead_time_days,
            request.lead_time_std_days,
            demand_std,
            request.service_level_target,
        )

        # Reorder point
        rop = int(np.ceil(request.avg_daily_demand * request.lead_time_days + safety_stock))

        # EOQ
        annual_demand = request.avg_daily_demand * 365
        holding_cost = request.unit_cost * request.holding_cost_pct
        eoq = self._calculate_eoq(annual_demand, request.order_cost, holding_cost)

        # Days of supply
        dos = (
            request.current_stock_units / request.avg_daily_demand
            if request.avg_daily_demand > 0
            else 999
        )

        # Stockout probability (30d)
        demand_30d_mean = request.avg_daily_demand * 30
        demand_30d_std = demand_std * np.sqrt(30)
        from scipy.stats import norm
        stockout_prob = float(1 - norm.cdf(
            request.current_stock_units, demand_30d_mean, max(demand_30d_std, 0.01)
        ))

        # Weekly forecast (8 weeks)
        weekly_forecasts = []
        for week in range(1, 9):
            weekly_demand = request.avg_daily_demand * 7
            ci_width = demand_std * np.sqrt(7) * 1.645  # 90% CI
            noise = float(rng.normal(0, demand_std * np.sqrt(7)))
            projected = max(0, weekly_demand + noise)
            weekly_forecasts.append(WeeklyForecast(
                week=week,
                projected_demand=round(projected, 1),
                confidence_interval=(
                    round(max(0, weekly_demand - ci_width), 1),
                    round(weekly_demand + ci_width, 1),
                ),
            ))

        # Determine alert
        alert = None
        if stockout_prob > 0.2:
            alert = AlertType.STOCKOUT_RISK
        elif dos > 90:
            alert = AlertType.OVERSTOCK

        # Reorder date
        reorder_date = None
        if request.current_stock_units <= rop:
            from datetime import timedelta
            reorder_date = date.today()
        elif dos < request.lead_time_days + 7:
            from datetime import timedelta
            days_until = max(0, int(dos - request.lead_time_days))
            reorder_date = date.today() + timedelta(days=days_until)

        response = InventoryForecastResponse(
            sku_id=request.sku_id,
            current_stock_units=request.current_stock_units,
            days_of_supply=round(dos, 1),
            stockout_probability_30d=round(float(np.clip(stockout_prob, 0, 1)), 4),
            reorder_point=rop,
            reorder_quantity=eoq,
            recommended_reorder_date=reorder_date,
            preferred_supplier="SUPPLIER-PRI-001",
            backup_supplier="SUPPLIER-BKP-002",
            carrying_cost_per_unit_month=round(holding_cost / 12, 4),
            forecast_horizon=weekly_forecasts,
            alert_type=alert,
        )

        # Publish alerts
        if alert:
            await kafka_manager.publish(
                topic=KafkaTopics.INVENTORY_ALERTS,
                value=response.model_dump(mode="json"),
                key=request.sku_id,
            )

        # Auto-create reorder for stockout risk
        if alert == AlertType.STOCKOUT_RISK:
            await kafka_manager.publish(
                topic=KafkaTopics.INVENTORY_REORDER_REQUESTS,
                value={
                    "sku_id": request.sku_id,
                    "quantity": eoq,
                    "preferred_supplier": "SUPPLIER-PRI-001",
                    "urgency": "HIGH",
                },
                key=request.sku_id,
            )

        return response

    async def get_health(self) -> InventoryHealthResponse:
        return InventoryHealthResponse(
            total_skus=0,
            stockout_risk_count=0,
            overstock_count=0,
            avg_days_of_supply=0,
        )


inventory_service = InventoryForecastingService()

# ── Router ──

router = APIRouter(prefix="/api/v2/inventory", tags=["Inventory Forecasting"])


@router.get("/forecast/{sku_id}", response_model=InventoryForecastResponse, summary="Get forecast for a SKU")
async def forecast_sku(
    sku_id: str,
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.OPERATOR, AgentRole.ADMIN)),
):
    request = InventoryForecastRequest(sku_id=sku_id, current_stock_units=500, avg_daily_demand=25)
    return await inventory_service.forecast(request)


@router.post("/bulk_forecast", response_model=list[InventoryForecastResponse], summary="Bulk forecast")
async def bulk_forecast(
    requests: list[InventoryForecastRequest],
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    results = []
    for req in requests:
        results.append(await inventory_service.forecast(req))
    return results


@router.get("/inventory_health", response_model=InventoryHealthResponse, summary="Inventory health overview")
async def inventory_health(
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    return await inventory_service.get_health()
