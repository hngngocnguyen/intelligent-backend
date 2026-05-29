from __future__ import annotations

from fastapi import APIRouter

from ..models.hf_model import get_huggingface_service
from ..schemas.requests import SymptomAnalysisRequest, SymptomAnalysisResponse

router = APIRouter(prefix="/analyze", tags=["huggingface"])


@router.post("/symptoms", response_model=SymptomAnalysisResponse)
def analyze_symptoms(payload: SymptomAnalysisRequest) -> SymptomAnalysisResponse:
    service = get_huggingface_service()
    return service.analyze(payload)
