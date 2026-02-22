"""
ProductionSchedulerAgent — FastAPI Router
REST: POST /optimize_schedule, POST /lock_job, GET /bottleneck_analysis
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.agents.production_scheduler.schemas import (
    BottleneckAnalysis,
    MachineCapability,
    ScheduleRequest,
    ScheduleResponse,
)
from src.agents.production_scheduler.service import production_scheduler_service
from src.core.auth import AgentRole, AuthUser, require_role
from src.core.observability import AGENT_REQUEST_COUNT, AGENT_REQUEST_LATENCY

router = APIRouter(prefix="/api/v2/scheduler", tags=["Production Scheduler"])


@router.post(
    "/optimize_schedule",
    response_model=ScheduleResponse,
    summary="Generate optimized production schedule",
)
async def optimize_schedule(
    request: ScheduleRequest,
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    """Run the constraint-satisfaction optimizer on the job queue."""
    import time

    start = time.monotonic()
    try:
        result = await production_scheduler_service.optimize_schedule(request)
        AGENT_REQUEST_COUNT.labels(
            agent_name="production_scheduler", endpoint="/optimize_schedule",
            method="POST", status_code="200",
        ).inc()
        return result
    except Exception as e:
        AGENT_REQUEST_COUNT.labels(
            agent_name="production_scheduler", endpoint="/optimize_schedule",
            method="POST", status_code="500",
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        AGENT_REQUEST_LATENCY.labels(
            agent_name="production_scheduler", endpoint="/optimize_schedule",
        ).observe(time.monotonic() - start)


@router.post(
    "/lock_job",
    summary="Lock a job to prevent rescheduling",
)
async def lock_job(
    job_id: str,
    user: AuthUser = Depends(require_role(AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    """Lock a job so it cannot be rescheduled by the optimizer."""
    return await production_scheduler_service.lock_job(job_id)


@router.get(
    "/bottleneck_analysis",
    response_model=list[BottleneckAnalysis],
    summary="Get bottleneck analysis for production floor",
)
async def bottleneck_analysis(
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    """Analyze bottleneck machines and provide recommendations."""
    return await production_scheduler_service.get_bottleneck_analysis([])
