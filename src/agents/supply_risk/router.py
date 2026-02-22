"""SupplyRiskAgent — Router"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.agents.supply_risk.schemas import RiskAssessmentResponse, SupplierRiskRequest
from src.agents.supply_risk.service import supply_risk_service
from src.core.auth import AgentRole, AuthUser, require_role

router = APIRouter(prefix="/api/v2/supply-risk", tags=["Supply Risk"])


@router.post("/risk_assessment", response_model=RiskAssessmentResponse, summary="Assess supplier risk")
async def assess_risk(
    request: SupplierRiskRequest,
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    return await supply_risk_service.assess_risk(request)
