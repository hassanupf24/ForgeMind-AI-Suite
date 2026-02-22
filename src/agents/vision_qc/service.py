"""
VisionQC_Agent — Service Layer
EfficientNet-B4 + YOLOv8 vision pipeline with GradCAM explainability.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np

from src.agents.vision_qc.schemas import (
    BatchReport,
    BoundingBox,
    DefectType,
    DetectedDefect,
    InspectionRequest,
    InspectionResponse,
    QCDecision,
    TIER_1_DEFECTS,
    TIER_2_DEFECTS,
    TIER_3_DEFECTS,
)
from src.core.cache import redis_cache
from src.core.kafka_manager import KafkaTopics, kafka_manager

logger = logging.getLogger(__name__)


class VisionQCService:
    """Computer vision QC engine with dual-model ensemble."""

    def __init__(self) -> None:
        self._model_version = "efficientnet-b4-v2.1"
        self._confidence_threshold = 0.85
        self._latency_sla_ms = 150
        self._batch_stats: dict[str, dict] = {}

    async def inspect(
        self,
        request: InspectionRequest,
        image_bytes: Optional[bytes] = None,
    ) -> InspectionResponse:
        """Run full vision inspection pipeline.

        Pipeline steps:
        1. Preprocessing (histogram equalization, denoising, normalization)
        2. EfficientNet-B4 classification
        3. YOLOv8 defect localization
        4. Ensemble confidence fusion
        5. GradCAM heatmap generation
        6. Decision + escalation logic
        """
        start_time = time.monotonic()

        # Validate image quality
        lighting_ok = self._check_lighting(image_bytes)
        if not lighting_ok:
            return InspectionResponse(
                product_id=request.product_id,
                batch_id=request.batch_id,
                qc_decision=QCDecision.HOLD,
                overall_confidence=0.0,
                inspection_latency_ms=int((time.monotonic() - start_time) * 1000),
                model_version=self._model_version,
                escalated_to_human=True,
                escalation_reason="IMAGING_FAULT: brightness/contrast outside calibration bounds",
            )

        # Simulate inference (production would load actual models)
        defects = self._run_inference(image_bytes)

        # Decision logic
        qc_decision, escalated, escalation_reason = self._make_decision(defects)

        latency_ms = int((time.monotonic() - start_time) * 1000)

        # SLA check
        if latency_ms > 200:
            logger.warning(
                "SLA breach: inspection latency %dms > 200ms for product %s",
                latency_ms,
                request.product_id,
            )

        response = InspectionResponse(
            inspection_id=uuid4(),
            product_id=request.product_id,
            batch_id=request.batch_id,
            timestamp=datetime.now(timezone.utc),
            qc_decision=qc_decision,
            defects_detected=defects,
            overall_confidence=self._compute_overall_confidence(defects),
            inspection_latency_ms=latency_ms,
            model_version=self._model_version,
            escalated_to_human=escalated,
            escalation_reason=escalation_reason,
        )

        # Publish events
        await self._publish_events(response)

        # Track batch stats
        self._update_batch_stats(request.batch_id, response)

        return response

    def _check_lighting(self, image_bytes: Optional[bytes]) -> bool:
        """Check if image brightness/contrast is within calibration bounds."""
        if image_bytes is None:
            return True  # No image = simulated pass
        # In production: compute mean brightness, check against [50, 220] range
        return True

    def _run_inference(self, image_bytes: Optional[bytes]) -> list[DetectedDefect]:
        """Run EfficientNet + YOLOv8 inference pipeline.

        Returns detected defects with bounding boxes and confidence scores.
        """
        # Simulated inference for demonstration
        # In production: load TensorRT-optimized models, run actual inference
        rng = np.random.default_rng()

        # Most products pass (95% pass rate simulation)
        if rng.random() > 0.05:
            return []

        # Generate random defect for demonstration
        defect_types = list(DefectType)
        defect_type = rng.choice(defect_types)
        confidence = float(rng.uniform(0.6, 0.99))

        tier = 3
        if defect_type in TIER_1_DEFECTS:
            tier = 1
        elif defect_type in TIER_2_DEFECTS:
            tier = 2

        return [
            DetectedDefect(
                defect_type=defect_type,
                severity_tier=tier,
                confidence=round(confidence, 4),
                bounding_box=BoundingBox(
                    x=round(float(rng.uniform(0.1, 0.8)), 3),
                    y=round(float(rng.uniform(0.1, 0.8)), 3),
                    w=round(float(rng.uniform(0.05, 0.3)), 3),
                    h=round(float(rng.uniform(0.05, 0.3)), 3),
                ),
                gradcam_url=f"/gradcam/{uuid4()}.png",
            )
        ]

    def _make_decision(
        self,
        defects: list[DetectedDefect],
    ) -> tuple[QCDecision, bool, Optional[str]]:
        """Apply decision logic based on defect severity and confidence."""
        if not defects:
            return QCDecision.PASS, False, None

        has_critical = any(d.severity_tier == 1 for d in defects)
        has_major = any(d.severity_tier == 2 for d in defects)
        low_confidence_fail = any(
            d.severity_tier <= 2 and d.confidence < self._confidence_threshold
            for d in defects
        )

        if has_critical:
            # Check confidence for auto-reject
            critical_defects = [d for d in defects if d.severity_tier == 1]
            if all(d.confidence >= self._confidence_threshold for d in critical_defects):
                return QCDecision.FAIL, False, None
            else:
                return QCDecision.HOLD, True, "CRITICAL defect with low confidence — human review required"

        if has_major:
            return QCDecision.HOLD, True, "Major defect detected — hold for review"

        if low_confidence_fail:
            return QCDecision.HOLD, True, "Low confidence detection — human review required"

        # Only minor defects
        return QCDecision.PASS, False, None

    def _compute_overall_confidence(self, defects: list[DetectedDefect]) -> float:
        """Compute ensemble confidence across all detections."""
        if not defects:
            return 0.99  # High confidence in PASS
        return round(float(np.mean([d.confidence for d in defects])), 4)

    async def _publish_events(self, response: InspectionResponse) -> None:
        """Publish QC events to Kafka."""
        event = response.model_dump(mode="json")

        await kafka_manager.publish(
            topic=KafkaTopics.FACTORY_QC_EVENTS,
            value=event,
            key=response.product_id,
        )

        if response.qc_decision == QCDecision.FAIL:
            await kafka_manager.publish(
                topic=KafkaTopics.FACTORY_QC_REJECTS,
                value=event,
                key=response.product_id,
            )

        if response.escalated_to_human:
            await kafka_manager.publish(
                topic=KafkaTopics.FACTORY_QC_ESCALATIONS,
                value=event,
                key=response.product_id,
            )

    def _update_batch_stats(self, batch_id: str, response: InspectionResponse) -> None:
        """Track per-batch inspection statistics."""
        if batch_id not in self._batch_stats:
            self._batch_stats[batch_id] = {
                "total": 0, "passed": 0, "failed": 0, "held": 0,
                "defect_counts": {},
            }

        stats = self._batch_stats[batch_id]
        stats["total"] += 1

        if response.qc_decision == QCDecision.PASS:
            stats["passed"] += 1
        elif response.qc_decision == QCDecision.FAIL:
            stats["failed"] += 1
        else:
            stats["held"] += 1

        for defect in response.defects_detected:
            dt = defect.defect_type.value
            stats["defect_counts"][dt] = stats["defect_counts"].get(dt, 0) + 1

    async def get_batch_report(self, batch_id: str) -> Optional[BatchReport]:
        """Generate a batch quality report."""
        stats = self._batch_stats.get(batch_id)
        if not stats:
            return None

        total = stats["total"]
        return BatchReport(
            batch_id=batch_id,
            total_inspected=total,
            passed=stats["passed"],
            failed=stats["failed"],
            held=stats["held"],
            defect_breakdown=stats["defect_counts"],
            batch_yield_pct=round(stats["passed"] / total * 100, 2) if total > 0 else 0.0,
        )


# Singleton
vision_qc_service = VisionQCService()
