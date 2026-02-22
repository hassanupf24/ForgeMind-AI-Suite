"""
ForgeMind AI Suite — Main FastAPI Application
Assembles all 15 agents into a unified API gateway.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.core.cache import redis_cache
from src.core.config import get_settings
from src.core.database import close_databases, init_databases
from src.core.kafka_manager import kafka_manager
from src.core.observability import (
    AGENT_HEALTH,
    instrument_fastapi,
    setup_observability,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup & shutdown hooks."""
    # ── Startup ──
    setup_observability()

    # Initialize databases
    try:
        await init_databases()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Database init skipped: %s", e)

    # Initialize Kafka producer
    try:
        await kafka_manager.start_producer()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Kafka init skipped: %s", e)

    # Initialize Redis
    try:
        await redis_cache.connect()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Redis init skipped: %s", e)

    # Set all agents as healthy
    agent_names = [
        "predictive_maintenance", "production_scheduler", "vision_qc",
        "supply_risk", "energy_optimization", "inventory_forecasting",
        "worker_safety", "process_analyzer", "root_cause_analysis",
        "demand_planning", "digital_twin", "supplier_performance",
        "waste_reduction", "reporting",
    ]
    for agent in agent_names:
        AGENT_HEALTH.labels(agent_name=agent).set(1)

    yield

    # ── Shutdown ──
    await kafka_manager.shutdown()
    await redis_cache.disconnect()
    await close_databases()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="ForgeMind AI Suite",
        version=settings.app_version,
        description=(
            "🏭 Advanced Manufacturing Intelligence Platform — "
            "15 autonomous AI agents orchestrating predictive maintenance, "
            "production scheduling, quality control, supply chain risk, "
            "energy optimization, worker safety, and more."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error Handlers ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    # ── Register Agent Routers ──
    from src.agents.predictive_maintenance.router import router as maintenance_router
    from src.agents.production_scheduler.router import router as scheduler_router
    from src.agents.vision_qc.router import router as qc_router
    from src.agents.supply_risk.router import router as supply_risk_router
    from src.agents.energy_optimization.agent import router as energy_router
    from src.agents.inventory_forecasting.agent import router as inventory_router
    from src.agents.worker_safety.agent import router as safety_router
    from src.agents.process_analyzer.agent import router as spc_router
    from src.agents.root_cause_analysis.agent import router as rca_router
    from src.agents.demand_planning.agent import router as demand_router
    from src.agents.digital_twin.agent import router as twin_router
    from src.agents.supplier_performance.agent import router as supplier_perf_router
    from src.agents.waste_reduction.agent import router as waste_router
    from src.agents.reporting.agent import router as reporting_router

    app.include_router(maintenance_router)
    app.include_router(scheduler_router)
    app.include_router(qc_router)
    app.include_router(supply_risk_router)
    app.include_router(energy_router)
    app.include_router(inventory_router)
    app.include_router(safety_router)
    app.include_router(spc_router)
    app.include_router(rca_router)
    app.include_router(demand_router)
    app.include_router(twin_router)
    app.include_router(supplier_perf_router)
    app.include_router(waste_router)
    app.include_router(reporting_router)

    # ── Health & Root ──
    @app.get("/", tags=["System"])
    async def root():
        return {
            "service": "ForgeMind AI Suite",
            "version": settings.app_version,
            "status": "operational",
            "agents": 15,
            "docs": "/docs",
        }

    @app.get("/health", tags=["System"])
    async def health():
        return {
            "status": "healthy",
            "version": settings.app_version,
            "environment": settings.app_env.value,
        }

    @app.get("/agents", tags=["System"])
    async def list_agents():
        return {
            "agents": [
                {"name": "PredictiveMaintenanceAgent", "prefix": "/api/v2/maintenance", "status": "active"},
                {"name": "ProductionSchedulerAgent", "prefix": "/api/v2/scheduler", "status": "active"},
                {"name": "VisionQC_Agent", "prefix": "/api/v2/qc", "status": "active"},
                {"name": "SupplyRiskAgent", "prefix": "/api/v2/supply-risk", "status": "active"},
                {"name": "EnergyOptimizationAgent", "prefix": "/api/v2/energy", "status": "active"},
                {"name": "InventoryForecastingAgent", "prefix": "/api/v2/inventory", "status": "active"},
                {"name": "WorkerSafetyAgent", "prefix": "/api/v2/safety", "status": "active"},
                {"name": "ProcessAnalyzerAgent", "prefix": "/api/v2/spc", "status": "active"},
                {"name": "RootCauseAnalysisAgent", "prefix": "/api/v2/rca", "status": "active"},
                {"name": "DemandPlanningAgent", "prefix": "/api/v2/demand", "status": "active"},
                {"name": "DigitalTwinAgent", "prefix": "/api/v2/digital-twin", "status": "active"},
                {"name": "SupplierPerformanceAgent", "prefix": "/api/v2/supplier-performance", "status": "active"},
                {"name": "WasteReductionAgent", "prefix": "/api/v2/waste", "status": "active"},
                {"name": "ReportingAgent", "prefix": "/api/v2/reports", "status": "active"},
            ],
        }

    # Instrument with OpenTelemetry
    instrument_fastapi(app)

    return app


app = create_app()
