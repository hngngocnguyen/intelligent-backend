from __future__ import annotations

from fastapi import APIRouter

from ..models.llm_model import get_llm_service
from ..schemas.requests import ExplanationRequest, MedicalExplanationResponse

router = APIRouter(prefix="/explain", tags=["llm"])


@router.post("", response_model=MedicalExplanationResponse)
def explain(payload: ExplanationRequest) -> MedicalExplanationResponse:
    service = get_llm_service()
    return service.generate(payload)
