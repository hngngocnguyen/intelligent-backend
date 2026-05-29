from __future__ import annotations

from fastapi import APIRouter

from ..models.classic_model import get_classic_service
from ..models.hf_model import get_huggingface_service
from ..models.llm_model import get_llm_service
from ..schemas.requests import FullPipelineRequest, FullPipelineResponse, PipelineConsensus
from ..utils.preprocessing import risk_level_from_probability

router = APIRouter(prefix="/full-pipeline", tags=["pipeline"])


@router.post("", response_model=FullPipelineResponse)
def full_pipeline(payload: FullPipelineRequest) -> FullPipelineResponse:
    classic_service = get_classic_service()
    hf_service = get_huggingface_service()
    llm_service = get_llm_service()

    classic_result = classic_service.predict(payload.clinical_data, model_version=payload.model_version)
    hf_result = hf_service.analyze(payload.to_symptom_request(), classic_probability=classic_result.probability)
    llm_result = llm_service.generate(payload.to_explanation_request(risk_probability=classic_result.probability))

    tabular_risk = risk_level_from_probability(classic_result.probability)
    hf_contradicts = hf_result.symptom_matches_prediction is False
    llm_contradicts = llm_result.risk_level != tabular_risk

    if hf_contradicts and llm_contradicts:
        agreement = "disagree"
    elif classic_result.risk_level == "low" and hf_result.top_category == "no significant symptoms":
        agreement = "all_agree"
    elif classic_result.risk_level == "high" and "diabetes" in hf_result.top_category.lower():
        agreement = "all_agree"
    elif hf_contradicts or llm_contradicts:
        agreement = "partial"
    else:
        agreement = "all_agree"

    consensus = PipelineConsensus(
        agreement=agreement,
        final_risk_level=llm_result.risk_level,
        cross_validation=(
            "symptoms align with the tabular prediction"
            if hf_result.symptom_matches_prediction
            else "mixed evidence across modalities"
        ),
    )

    return FullPipelineResponse(
        input=payload,
        results=FullPipelineResponse.Results(
            classic_model=classic_result,
            huggingface=hf_result,
            llm=llm_result,
        ),
        consensus=consensus,
    )
