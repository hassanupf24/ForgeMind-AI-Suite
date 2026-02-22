"""
WasteReductionAgent — Schemas, Service & Router
Lean + Theory of Constraints: Muda, Mura, Muri analysis. Kaizen proposals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.core.auth import AgentRole, AuthUser, require_role


class WasteCategory(str, Enum):
    MUDA = "MUDA"   # Waste
    MURA = "MURA"   # Unevenness
    MURI = "MURI"   # Overburden


class WasteType(str, Enum):
    SCRAP = "SCRAP"
    REWORK = "REWORK"
    OVERPRODUCTION = "OVERPRODUCTION"
    WAITING = "WAITING"
    MOTION = "MOTION"
    TRANSPORTATION = "TRANSPORTATION"
    OVERPROCESSING = "OVERPROCESSING"
    INVENTORY = "INVENTORY"


class WasteItem(BaseModel):
    waste_type: WasteType
    category: WasteCategory
    quantity_units: float = 0.0
    cost_usd: float = 0.0
    co2_kg: float = 0.0
    source_process: str = ""
    shift_id: str = ""


class KaizenProposal(BaseModel):
    title: str
    description: str
    target_waste_type: WasteType
    effort: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    impact: str = "MEDIUM"
    estimated_savings_usd: float = 0.0
    estimated_scrap_reduction_pct: float = 0.0
    priority_score: float = 0.0  # impact/effort ratio


class WasteAnalysisResponse(BaseModel):
    period: str
    total_scrap_cost_usd: float = 0.0
    total_rework_cost_usd: float = 0.0
    total_overproduction_cost_usd: float = 0.0
    total_co2_kg: float = 0.0
    waste_items: list[WasteItem] = Field(default_factory=list)
    top_3_categories: list[dict] = Field(default_factory=list)
    kaizen_proposals: list[KaizenProposal] = Field(default_factory=list)
    scrap_rate_pct: float = 0.0
    target_scrap_reduction_pct: float = 15.0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WasteReductionService:

    async def analyze(self, period: str = "shift") -> WasteAnalysisResponse:
        rng = np.random.default_rng()

        waste_items = []
        for wt in [WasteType.SCRAP, WasteType.REWORK, WasteType.OVERPRODUCTION]:
            cost = round(float(rng.uniform(500, 5000)), 2)
            waste_items.append(WasteItem(
                waste_type=wt,
                category=WasteCategory.MUDA,
                quantity_units=round(float(rng.uniform(10, 200)), 0),
                cost_usd=cost,
                co2_kg=round(cost * 0.3, 1),
                source_process=f"PROCESS-{rng.integers(1, 10):02d}",
                shift_id=f"SHIFT-{rng.choice(['A', 'B', 'C'])}",
            ))

        total_scrap = sum(w.cost_usd for w in waste_items if w.waste_type == WasteType.SCRAP)
        total_rework = sum(w.cost_usd for w in waste_items if w.waste_type == WasteType.REWORK)
        total_overprod = sum(w.cost_usd for w in waste_items if w.waste_type == WasteType.OVERPRODUCTION)

        proposals = [
            KaizenProposal(
                title="Implement poka-yoke on station 3",
                description="Install mistake-proofing fixture to prevent misalignment defects",
                target_waste_type=WasteType.SCRAP,
                effort="LOW",
                impact="HIGH",
                estimated_savings_usd=round(total_scrap * 0.3, 2),
                estimated_scrap_reduction_pct=30,
                priority_score=9.0,
            ),
            KaizenProposal(
                title="Reduce batch size on line 2",
                description="Implement single-piece flow to reduce overproduction and WIP",
                target_waste_type=WasteType.OVERPRODUCTION,
                effort="MEDIUM",
                impact="HIGH",
                estimated_savings_usd=round(total_overprod * 0.4, 2),
                estimated_scrap_reduction_pct=20,
                priority_score=7.5,
            ),
            KaizenProposal(
                title="Standardize rework procedure",
                description="Create standard work for rework operations to reduce touch time",
                target_waste_type=WasteType.REWORK,
                effort="LOW",
                impact="MEDIUM",
                estimated_savings_usd=round(total_rework * 0.2, 2),
                estimated_scrap_reduction_pct=10,
                priority_score=6.0,
            ),
        ]

        return WasteAnalysisResponse(
            period=period,
            total_scrap_cost_usd=round(total_scrap, 2),
            total_rework_cost_usd=round(total_rework, 2),
            total_overproduction_cost_usd=round(total_overprod, 2),
            total_co2_kg=round(sum(w.co2_kg for w in waste_items), 1),
            waste_items=waste_items,
            top_3_categories=[
                {"category": "SCRAP", "cost": round(total_scrap, 2)},
                {"category": "REWORK", "cost": round(total_rework, 2)},
                {"category": "OVERPRODUCTION", "cost": round(total_overprod, 2)},
            ],
            kaizen_proposals=proposals,
            scrap_rate_pct=round(float(rng.uniform(1, 5)), 2),
        )


waste_service = WasteReductionService()

router = APIRouter(prefix="/api/v2/waste", tags=["Waste Reduction"])


@router.get("/analysis", response_model=WasteAnalysisResponse, summary="Get waste analysis")
async def waste_analysis(
    period: str = "shift",
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    return await waste_service.analyze(period)
