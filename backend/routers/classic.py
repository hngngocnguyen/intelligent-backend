from __future__ import annotations

from fastapi import APIRouter, Query

from ..models.classic_model import get_classic_service
from ..schemas.requests import ClinicalData, ClassicPredictionResponse

router = APIRouter(prefix="/predict", tags=["classic"])


@router.post("/tabular", response_model=ClassicPredictionResponse)
def predict_tabular(
    payload: ClinicalData,
    model_version: str = Query(default="mlp_v3", pattern="^mlp_v[123]$"),
) -> ClassicPredictionResponse:
    service = get_classic_service()
    return service.predict(payload, model_version=model_version)
