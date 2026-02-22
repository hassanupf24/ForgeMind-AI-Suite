"""
PredictiveMaintenanceAgent — FastAPI Router
REST endpoints: POST /predict, GET /rul/{machine_id}, GET /health_dashboard
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.agents.predictive_maintenance.schemas import (
    HealthDashboardResponse,
    MachineMetadata,
    PredictionResponse,
    RULResponse,
    TelemetryReading,
)
from src.agents.predictive_maintenance.service import predictive_maintenance_service
from src.core.auth import AgentRole, AuthUser, require_role
from src.core.observability import AGENT_REQUEST_COUNT, AGENT_REQUEST_LATENCY

router = APIRouter(
    prefix="/api/v2/maintenance",
    tags=["Predictive Maintenance"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run failure prediction on telemetry data",
    description="Analyze a telemetry reading through the 5-step reasoning protocol "
                "and return failure probabilities, RUL, and recommended actions.",
)
async def predict(
    reading: TelemetryReading,
    metadata: MachineMetadata | None = None,
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    """Execute the full predictive maintenance pipeline."""
    import time

    start = time.monotonic()
    try:
        result = await predictive_maintenance_service.predict(reading, metadata)
        AGENT_REQUEST_COUNT.labels(
            agent_name="predictive_maintenance",
            endpoint="/predict",
            method="POST",
            status_code="200",
        ).inc()
        return result
    except Exception as e:
        AGENT_REQUEST_COUNT.labels(
            agent_name="predictive_maintenance",
            endpoint="/predict",
            method="POST",
            status_code="500",
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.monotonic() - start
        AGENT_REQUEST_LATENCY.labels(
            agent_name="predictive_maintenance",
            endpoint="/predict",
        ).observe(latency)


@router.get(
    "/rul/{machine_id}",
    response_model=RULResponse,
    summary="Get Remaining Useful Life for a machine",
)
async def get_rul(
    machine_id: str,
    user: AuthUser = Depends(require_role(
        AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.OPERATOR, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    """Retrieve the latest RUL estimate from cache."""
    result = await predictive_maintenance_service.get_rul(machine_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No RUL data available for machine {machine_id}",
        )
    return result


@router.get(
    "/health_dashboard",
    response_model=HealthDashboardResponse,
    summary="Get aggregated health dashboard data",
)
async def health_dashboard(
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    """Retrieve the aggregated health dashboard."""
    # In production, this aggregates from Redis/DB
    return HealthDashboardResponse(
        machines=[],
        overall_health_score=0.95,
        critical_alerts=0,
        pending_work_orders=0,
    )
