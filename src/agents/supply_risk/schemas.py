"""
SupplyRiskAgent — Schemas & Service
Multi-dimensional supply chain risk intelligence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskBand(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    CRITICAL = "CRITICAL"


class RiskTrend(str, Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"


class ActionUrgency(str, Enum):
    IMMEDIATE = "IMMEDIATE"
    THIRTY_DAYS = "30_DAYS"
    QUARTERLY = "QUARTERLY"


class RiskFactor(BaseModel):
    factor: str
    impact: str
    evidence: str


class AlternativeSupplier(BaseModel):
    supplier_id: str
    fit_score: float = Field(..., ge=0.0, le=1.0)


class RecommendedRiskAction(BaseModel):
    action: str
    urgency: ActionUrgency
    alternative_suppliers: list[AlternativeSupplier] = Field(default_factory=list)


class DimensionScores(BaseModel):
    delivery_performance: float = Field(0.0, ge=0, le=100)
    quality_rejection_rate: float = Field(0.0, ge=0, le=100)
    financial_stability: float = Field(0.0, ge=0, le=100)
    geopolitical_exposure: float = Field(0.0, ge=0, le=100)
    single_source_dependency: float = Field(0.0, ge=0, le=100)
    lead_time_volatility: float = Field(0.0, ge=0, le=100)
    communication_responsiveness: float = Field(0.0, ge=0, le=100)


class SupplierRiskRequest(BaseModel):
    supplier_id: str
    delivery_on_time_pct: float = Field(95.0, ge=0, le=100)
    quality_ppm: float = Field(500, ge=0)
    financial_rating: str = "BBB"
    country_code: str = "US"
    is_sole_source: bool = False
    avg_lead_time_days: float = Field(14, ge=0)
    lead_time_std_days: float = Field(2, ge=0)
    crm_response_hours: float = Field(4, ge=0)
    bom_spend_pct: float = Field(5.0, ge=0, le=100)


class RiskAssessmentResponse(BaseModel):
    supplier_id: str
    assessment_date: datetime = Field(default_factory=lambda: datetime.utcnow())
    composite_risk_index: float = Field(..., ge=0, le=100)
    risk_band: RiskBand
    dimension_scores: DimensionScores
    trend: RiskTrend
    top_risk_factors: list[RiskFactor] = Field(default_factory=list)
    recommended_actions: list[RecommendedRiskAction] = Field(default_factory=list)
    portfolio_exposure_pct: float = Field(0.0, ge=0, le=100)
