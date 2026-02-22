"""
ForgeMind AI Suite — OpenTelemetry Observability Setup
Distributed tracing (Jaeger), metrics (Prometheus), structured logging.
"""

from __future__ import annotations

import logging
import sys

import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram, Info

from src.core.config import get_settings

settings = get_settings()

# ── Prometheus Metrics ──
AGENT_REQUEST_COUNT = Counter(
    "forgemind_agent_requests_total",
    "Total agent API requests",
    ["agent_name", "endpoint", "method", "status_code"],
)

AGENT_REQUEST_LATENCY = Histogram(
    "forgemind_agent_request_latency_seconds",
    "Agent API request latency",
    ["agent_name", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

AGENT_HEALTH = Gauge(
    "forgemind_agent_health",
    "Agent health status (1=healthy, 0=unhealthy)",
    ["agent_name"],
)

MODEL_INFERENCE_LATENCY = Histogram(
    "forgemind_model_inference_latency_seconds",
    "ML model inference latency",
    ["agent_name", "model_name"],
    buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0],
)

KAFKA_MESSAGE_COUNT = Counter(
    "forgemind_kafka_messages_total",
    "Total Kafka messages processed",
    ["topic", "direction"],  # direction: produced/consumed
)

DATA_QUALITY_SCORE = Gauge(
    "forgemind_data_quality_score",
    "Data quality score for incoming telemetry",
    ["machine_id"],
)

APP_INFO = Info(
    "forgemind_app",
    "ForgeMind application information",
)


def setup_tracing() -> None:
    """Configure OpenTelemetry distributed tracing with Jaeger exporter."""
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.app_env.value,
        }
    )

    tracer_provider = TracerProvider(resource=resource)

    jaeger_exporter = JaegerExporter(
        collector_endpoint=settings.jaeger_endpoint,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

    trace.set_tracer_provider(tracer_provider)


def setup_logging() -> None:
    """Configure structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if settings.is_production else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging to route through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )


def instrument_fastapi(app) -> None:
    """Add OpenTelemetry instrumentation to FastAPI."""
    FastAPIInstrumentor.instrument_app(app)


def setup_observability() -> None:
    """Initialize all observability components."""
    setup_logging()
    setup_tracing()

    APP_INFO.info(
        {
            "version": settings.app_version,
            "environment": settings.app_env.value,
        }
    )
