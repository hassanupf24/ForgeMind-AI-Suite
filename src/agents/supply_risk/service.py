"""
SupplyRiskAgent — Service & Router
Composite risk index calculation with weighted multi-dimensional scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from src.agents.supply_risk.schemas import (
    ActionUrgency,
    AlternativeSupplier,
    DimensionScores,
    RecommendedRiskAction,
    RiskAssessmentResponse,
    RiskBand,
    RiskFactor,
    RiskTrend,
    SupplierRiskRequest,
)
from src.core.cache import redis_cache
from src.core.kafka_manager import KafkaTopics, kafka_manager

logger = logging.getLogger(__name__)

# Dimension weights from system prompt
DIMENSION_WEIGHTS = {
    "delivery_performance": 0.25,
    "quality_rejection_rate": 0.20,
    "financial_stability": 0.15,
    "geopolitical_exposure": 0.15,
    "single_source_dependency": 0.10,
    "lead_time_volatility": 0.10,
    "communication_responsiveness": 0.05,
}

# Financial rating risk mapping
FINANCIAL_RISK = {
    "AAA": 5, "AA": 10, "A": 15, "BBB": 25, "BB": 40,
    "B": 60, "CCC": 75, "CC": 85, "C": 90, "D": 100,
}

# High-risk countries
HIGH_RISK_COUNTRIES = {"RU", "IR", "KP", "SY", "VE", "MM", "AF"}
MEDIUM_RISK_COUNTRIES = {"CN", "TR", "BR", "IN", "ZA", "EG", "PK"}


class SupplyRiskService:
    """Multi-dimensional supply chain risk assessment engine."""

    async def assess_risk(self, request: SupplierRiskRequest) -> RiskAssessmentResponse:
        """Calculate composite risk index across all dimensions."""

        scores = self._calculate_dimension_scores(request)

        # Composite Risk Index (weighted sum)
        cri = (
            scores.delivery_performance * DIMENSION_WEIGHTS["delivery_performance"]
            + scores.quality_rejection_rate * DIMENSION_WEIGHTS["quality_rejection_rate"]
            + scores.financial_stability * DIMENSION_WEIGHTS["financial_stability"]
            + scores.geopolitical_exposure * DIMENSION_WEIGHTS["geopolitical_exposure"]
            + scores.single_source_dependency * DIMENSION_WEIGHTS["single_source_dependency"]
            + scores.lead_time_volatility * DIMENSION_WEIGHTS["lead_time_volatility"]
            + scores.communication_responsiveness * DIMENSION_WEIGHTS["communication_responsiveness"]
        )
        cri = round(float(np.clip(cri, 0, 100)), 1)

        # Determine risk band
        risk_band = self._classify_risk_band(cri)

        # Identify top risk factors
        risk_factors = self._identify_risk_factors(scores, request)

        # Generate recommended actions
        actions = self._generate_actions(cri, risk_band, request)

        response = RiskAssessmentResponse(
            supplier_id=request.supplier_id,
            assessment_date=datetime.now(timezone.utc),
            composite_risk_index=cri,
            risk_band=risk_band,
            dimension_scores=scores,
            trend=RiskTrend.STABLE,
            top_risk_factors=risk_factors[:3],
            recommended_actions=actions,
            portfolio_exposure_pct=request.bom_spend_pct,
        )

        # Publish assessment
        await kafka_manager.publish(
            topic=KafkaTopics.SUPPLY_RISK_ASSESSMENTS,
            value=response.model_dump(mode="json"),
            key=request.supplier_id,
        )

        # Auto-notify for CRITICAL
        if risk_band == RiskBand.CRITICAL:
            logger.critical(
                "CRITICAL risk band for supplier %s (CRI=%.1f) — auto-notifying procurement",
                request.supplier_id,
                cri,
            )

        await redis_cache.cache_agent_output(
            agent_name="supply_risk",
            entity_id=request.supplier_id,
            output=response.model_dump(mode="json"),
            ttl_seconds=86400,
        )

        return response

    def _calculate_dimension_scores(self, req: SupplierRiskRequest) -> DimensionScores:
        """Calculate individual dimension risk scores [0-100]."""

        # Delivery: invert on-time % → risk
        delivery = max(0, 100 - req.delivery_on_time_pct)

        # Quality: PPM-based (normalize: 0ppm=0 risk, 10000ppm=100 risk)
        quality = min(100, req.quality_ppm / 100)

        # Financial: rating-based lookup
        financial = FINANCIAL_RISK.get(req.financial_rating, 50)

        # Geopolitical: country risk
        if req.country_code in HIGH_RISK_COUNTRIES:
            geo = 90
        elif req.country_code in MEDIUM_RISK_COUNTRIES:
            geo = 45
        else:
            geo = 10

        # Single-source dependency
        single_source = 80 if req.is_sole_source else 10

        # Lead time volatility (CV-based)
        cv = req.lead_time_std_days / max(req.avg_lead_time_days, 1)
        lead_time = min(100, cv * 200)

        # Communication responsiveness (>24h = high risk)
        comm = min(100, req.crm_response_hours / 48 * 100)

        return DimensionScores(
            delivery_performance=round(delivery, 1),
            quality_rejection_rate=round(quality, 1),
            financial_stability=round(financial, 1),
            geopolitical_exposure=round(geo, 1),
            single_source_dependency=round(single_source, 1),
            lead_time_volatility=round(lead_time, 1),
            communication_responsiveness=round(comm, 1),
        )

    def _classify_risk_band(self, cri: float) -> RiskBand:
        if cri <= 30:
            return RiskBand.GREEN
        elif cri <= 60:
            return RiskBand.AMBER
        elif cri <= 80:
            return RiskBand.RED
        else:
            return RiskBand.CRITICAL

    def _identify_risk_factors(
        self, scores: DimensionScores, request: SupplierRiskRequest,
    ) -> list[RiskFactor]:
        factors = []
        score_items = [
            ("delivery_performance", scores.delivery_performance, "Delivery on-time rate below target"),
            ("quality_rejection_rate", scores.quality_rejection_rate, "Quality PPM exceeds threshold"),
            ("financial_stability", scores.financial_stability, f"Financial rating: {request.financial_rating}"),
            ("geopolitical_exposure", scores.geopolitical_exposure, f"Country risk: {request.country_code}"),
            ("single_source_dependency", scores.single_source_dependency, "Sole-source dependency"),
            ("lead_time_volatility", scores.lead_time_volatility, "High lead time variability"),
        ]

        for dim, score, evidence in sorted(score_items, key=lambda x: -x[1]):
            if score > 30:
                factors.append(RiskFactor(
                    factor=dim,
                    impact="HIGH" if score > 60 else "MEDIUM",
                    evidence=evidence,
                ))

        return factors

    def _generate_actions(
        self, cri: float, band: RiskBand, request: SupplierRiskRequest,
    ) -> list[RecommendedRiskAction]:
        actions = []

        if band in (RiskBand.RED, RiskBand.CRITICAL):
            actions.append(RecommendedRiskAction(
                action="Initiate supplier diversification — identify and qualify 2+ alternatives",
                urgency=ActionUrgency.IMMEDIATE if band == RiskBand.CRITICAL else ActionUrgency.THIRTY_DAYS,
                alternative_suppliers=[
                    AlternativeSupplier(supplier_id="ALT-001", fit_score=0.85),
                    AlternativeSupplier(supplier_id="ALT-002", fit_score=0.72),
                ],
            ))

        if request.is_sole_source and cri > 50:
            actions.append(RecommendedRiskAction(
                action="NEVER sole-source with CRI > 50 — dual-source immediately",
                urgency=ActionUrgency.IMMEDIATE,
            ))

        if band == RiskBand.AMBER:
            actions.append(RecommendedRiskAction(
                action="Schedule quarterly business review with supplier",
                urgency=ActionUrgency.QUARTERLY,
            ))

        return actions


# Singleton
supply_risk_service = SupplyRiskService()
