"""
ForgeMind AI Suite — Kafka Producer & Consumer Management
Handles async Kafka message production and consumption across all agent topics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Kafka Topic Registry ──
class KafkaTopics:
    """Central registry of all Kafka topics used across the suite."""

    # Telemetry & Maintenance
    FACTORY_TELEMETRY = "factory.telemetry"
    FACTORY_TELEMETRY_DLQ = "factory.telemetry.dlq"
    FACTORY_TELEMETRY_DEGRADED = "factory.telemetry.degraded"
    FACTORY_MAINTENANCE_ALERTS = "factory.maintenance.alerts"
    FACTORY_MAINTENANCE_COMPLETED = "factory.maintenance.completed"
    FACTORY_WORKORDER_REQUESTS = "factory.workorder.requests"
    FACTORY_HEALTH_SCORES = "factory.health.scores"

    # Production Schedule
    FACTORY_SCHEDULE_CURRENT = "factory.schedule.current"
    FACTORY_SCHEDULE_DELTA = "factory.schedule.delta"

    # Quality Control
    FACTORY_QC_EVENTS = "factory.qc.events"
    FACTORY_QC_REJECTS = "factory.qc.rejects"
    FACTORY_QC_ESCALATIONS = "factory.qc.escalations"
    FACTORY_LINE_TRIGGER = "factory.line.trigger"

    # Inventory
    INVENTORY_ALERTS = "inventory.alerts"
    INVENTORY_REORDER_REQUESTS = "inventory.reorder.requests"
    INVENTORY_TRANSACTIONS = "inventory.transactions"

    # Supply Chain
    SUPPLY_RISK_ASSESSMENTS = "supply.risk.assessments"
    SUPPLY_DELIVERIES = "supply.deliveries"
    INVENTORY_PARTS_SNAPSHOT = "inventory.parts.snapshot"

    # Energy
    SMART_METERS_READINGS = "smart_meters.readings"
    GRID_DEMAND_RESPONSE = "grid.demand_response.events"
    ENERGY_RECOMMENDATIONS = "energy.recommendations"
    ENERGY_ALERTS = "energy.alerts"

    # Safety
    SAFETY_EVENTS = "safety.events"
    SAFETY_EMERGENCY = "safety.emergency"
    SAFETY_COMPLIANCE_REPORTS = "safety.compliance.reports"

    # Process Control
    FACTORY_MEASUREMENTS_STREAM = "factory.measurements.stream"
    PROCESS_CONTROL_ALERTS = "process.control.alerts"
    QUALITY_SPC_REPORTS = "quality.spc.reports"

    # Demand
    DEMAND_FORECAST_PUBLISHED = "demand.forecast.published"
    DEMAND_ALERTS = "demand.alerts"

    # Production Completions
    PRODUCTION_COMPLETIONS = "production.completions"

    # Sustainability
    SUSTAINABILITY_DASHBOARD = "sustainability.dashboard.feed"

    # Weather
    WEATHER_FORECAST = "weather.forecast"

    # Access Control
    ACCESS_CONTROL_EVENTS = "access_control.events"


class KafkaManager:
    """Manages Kafka producer and consumer lifecycle."""

    def __init__(self) -> None:
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: dict[str, AIOKafkaConsumer] = {}

    async def start_producer(self) -> None:
        """Initialize and start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            enable_idempotence=True,
            max_request_size=10_485_760,  # 10 MB
        )
        await self._producer.start()
        logger.info("Kafka producer started → %s", settings.kafka_bootstrap_servers)

    async def stop_producer(self) -> None:
        """Gracefully stop the Kafka producer."""
        if self._producer:
            await self._producer.stop()
            logger.info("Kafka producer stopped")

    async def publish(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[list[tuple[str, bytes]]] = None,
    ) -> None:
        """Publish a message to a Kafka topic."""
        if not self._producer:
            raise RuntimeError("Kafka producer not initialized. Call start_producer() first.")
        await self._producer.send_and_wait(
            topic=topic,
            value=value,
            key=key,
            headers=headers,
        )
        logger.debug("Published to %s: key=%s", topic, key)

    async def create_consumer(
        self,
        topics: list[str],
        group_id: Optional[str] = None,
        auto_offset_reset: str = "latest",
    ) -> AIOKafkaConsumer:
        """Create and start a Kafka consumer for the given topics."""
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=group_id or settings.kafka_group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=settings.kafka_enable_auto_commit,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        await consumer.start()
        consumer_id = f"consumer-{'-'.join(topics)}"
        self._consumers[consumer_id] = consumer
        logger.info("Kafka consumer started for topics: %s", topics)
        return consumer

    async def consume(
        self,
        topics: list[str],
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
        group_id: Optional[str] = None,
    ) -> None:
        """Start consuming messages and route to handler."""
        consumer = await self.create_consumer(topics, group_id=group_id)
        try:
            async for message in consumer:
                try:
                    await handler(message.value)
                except Exception as e:
                    logger.error(
                        "Error processing message from %s: %s",
                        message.topic,
                        e,
                        exc_info=True,
                    )
                    # Route to DLQ
                    await self.publish(
                        topic=f"{message.topic}.dlq",
                        value={
                            "original_topic": message.topic,
                            "original_value": message.value,
                            "error": str(e),
                        },
                        key=message.key.decode("utf-8") if message.key else None,
                    )
        finally:
            await consumer.stop()

    async def stop_all_consumers(self) -> None:
        """Stop all active consumers."""
        for consumer_id, consumer in self._consumers.items():
            await consumer.stop()
            logger.info("Stopped consumer: %s", consumer_id)
        self._consumers.clear()

    async def shutdown(self) -> None:
        """Full shutdown of all Kafka connections."""
        await self.stop_all_consumers()
        await self.stop_producer()


# Singleton instance
kafka_manager = KafkaManager()
