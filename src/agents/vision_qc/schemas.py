"""
VisionQC_Agent — Pydantic Schemas
Computer vision defect detection contracts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class QCDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    HOLD = "HOLD"


class DefectType(str, Enum):
    # Tier 1 — Critical (auto-reject)
    CRACK = "CRACK"
    FRACTURE = "FRACTURE"
    DELAMINATION = "DELAMINATION"
    MISSING_COMPONENT = "MISSING_COMPONENT"
    DIMENSIONAL_OUT_OF_SPEC = "DIMENSIONAL_OUT_OF_SPEC"
    # Tier 2 — Major (hold for review)
    SCRATCH_DEEP = "SCRATCH_DEEP"
    DENT = "DENT"
    DISCOLORATION_SEVERE = "DISCOLORATION_SEVERE"
    WELD_POROSITY = "WELD_POROSITY"
    # Tier 3 — Minor (pass with log)
    SCRATCH_SURFACE = "SCRATCH_SURFACE"
    COSMETIC_MARK = "COSMETIC_MARK"


TIER_1_DEFECTS = {DefectType.CRACK, DefectType.FRACTURE, DefectType.DELAMINATION,
                  DefectType.MISSING_COMPONENT, DefectType.DIMENSIONAL_OUT_OF_SPEC}
TIER_2_DEFECTS = {DefectType.SCRATCH_DEEP, DefectType.DENT,
                  DefectType.DISCOLORATION_SEVERE, DefectType.WELD_POROSITY}
TIER_3_DEFECTS = {DefectType.SCRATCH_SURFACE, DefectType.COSMETIC_MARK}


class BoundingBox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)


class InspectionRequest(BaseModel):
    product_id: str
    batch_id: str
    line_id: str
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    image_path: Optional[str] = None  # For file-based input


class DetectedDefect(BaseModel):
    defect_type: DefectType
    severity_tier: int = Field(..., ge=1, le=3)
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox
    gradcam_url: str = ""


class InspectionResponse(BaseModel):
    inspection_id: UUID = Field(default_factory=uuid4)
    product_id: str
    batch_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    qc_decision: QCDecision
    defects_detected: list[DetectedDefect] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    inspection_latency_ms: int = 0
    model_version: str = "efficientnet-b4-v2.1"
    escalated_to_human: bool = False
    escalation_reason: Optional[str] = None


class BatchReport(BaseModel):
    batch_id: str
    total_inspected: int = 0
    passed: int = 0
    failed: int = 0
    held: int = 0
    defect_breakdown: dict[str, int] = Field(default_factory=dict)
    batch_yield_pct: float = 0.0
    generated_at: datetime = Field(default_factory=lambda: datetime.utcnow())
