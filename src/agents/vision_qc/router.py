"""
VisionQC_Agent — FastAPI Router
REST: POST /inspect, GET /batch_report/{batch_id}
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.agents.vision_qc.schemas import BatchReport, InspectionRequest, InspectionResponse
from src.agents.vision_qc.service import vision_qc_service
from src.core.auth import AgentRole, AuthUser, require_role

router = APIRouter(prefix="/api/v2/qc", tags=["Vision QC"])


@router.post(
    "/inspect",
    response_model=InspectionResponse,
    summary="Run visual quality inspection on a product image",
)
async def inspect(
    product_id: str = Form(...),
    batch_id: str = Form(...),
    line_id: str = Form(...),
    camera_id: str = Form(...),
    image: UploadFile = File(None),
    user: AuthUser = Depends(require_role(AgentRole.AGENT, AgentRole.ENGINEER, AgentRole.ADMIN)),
):
    """Execute the full vision QC pipeline on an uploaded image."""
    image_bytes = None
    if image:
        image_bytes = await image.read()

    request = InspectionRequest(
        product_id=product_id,
        batch_id=batch_id,
        line_id=line_id,
        camera_id=camera_id,
    )

    return await vision_qc_service.inspect(request, image_bytes)


@router.get(
    "/batch_report/{batch_id}",
    response_model=BatchReport,
    summary="Get QC report for a production batch",
)
async def batch_report(
    batch_id: str,
    user: AuthUser = Depends(require_role(
        AgentRole.OPERATOR, AgentRole.ENGINEER, AgentRole.VIEWER, AgentRole.ADMIN
    )),
):
    """Get aggregated inspection results for a batch."""
    report = await vision_qc_service.get_batch_report(batch_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No inspection data for batch {batch_id}",
        )
    return report
