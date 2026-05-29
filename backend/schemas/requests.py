from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "moderate", "high"]


class ClinicalData(BaseModel):
    pregnancies: int = Field(ge=0, le=25)
    glucose: float = Field(ge=0)
    blood_pressure: float = Field(ge=0)
    skin_thickness: float = Field(ge=0)
    insulin: float = Field(ge=0)
    bmi: float = Field(ge=0)
    diabetes_pedigree_function: float = Field(ge=0)
    age: int = Field(ge=1, le=120)


class SymptomAnalysisRequest(BaseModel):
    symptoms_text: str = Field(min_length=2, max_length=4000)
    candidate_labels: list[str] | None = None


class ExplanationRequest(BaseModel):
    clinical_data: ClinicalData
    risk_probability: float = Field(ge=0.0, le=1.0)
    symptoms_text: str = Field(min_length=2, max_length=4000)
    symptom_analysis: dict[str, Any] | None = None
    model_version: str = "mlp_v3"


class FullPipelineRequest(BaseModel):
    clinical_data: ClinicalData
    symptoms_text: str = Field(min_length=2, max_length=4000)
    model_version: str = Field(default="mlp_v3", pattern="^mlp_v[123]$")
    candidate_labels: list[str] | None = None

    def to_symptom_request(self) -> SymptomAnalysisRequest:
        return SymptomAnalysisRequest(
            symptoms_text=self.symptoms_text,
            candidate_labels=self.candidate_labels,
        )

    def to_explanation_request(
        self,
        risk_probability: float,
        symptom_analysis: dict[str, Any] | None = None,
    ) -> ExplanationRequest:
        return ExplanationRequest(
            clinical_data=self.clinical_data,
            risk_probability=risk_probability,
            symptoms_text=self.symptoms_text,
            symptom_analysis=symptom_analysis,
            model_version=self.model_version,
        )


class SymptomScore(BaseModel):
    category: str
    score: float


class ClassicPredictionResponse(BaseModel):
    model_version: str
    prediction: str
    probability: float
    risk_level: RiskLevel
    inference_time_ms: int
    feature_summary: dict[str, Any]


class SymptomAnalysisResponse(BaseModel):
    model: str
    top_category: str
    confidence: float
    all_scores: dict[str, float]
    top_categories: list[SymptomScore]
    symptom_matches_prediction: bool | None = None
    inference_time_ms: int
    mode: str


class MedicalExplanationResponse(BaseModel):
    model: str
    mode: str
    risk_level: RiskLevel
    summary: str
    recommendations: list[str]
    urgency: Literal["routine", "soon", "immediate"]
    disclaimer: str
    inference_time_ms: int


class PipelineConsensus(BaseModel):
    agreement: Literal["all_agree", "partial", "disagree"]
    final_risk_level: RiskLevel
    cross_validation: str


class HealthResponse(BaseModel):
    status: str
    models: dict[str, Any]


class ModelInfoResponse(BaseModel):
    classic_model: dict[str, Any]
    huggingface_model: dict[str, Any]
    llm_model: dict[str, Any]


class FullPipelineResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class Results(BaseModel):
        classic_model: ClassicPredictionResponse
        huggingface: SymptomAnalysisResponse
        llm: MedicalExplanationResponse

    input: FullPipelineRequest
    results: Results
    consensus: PipelineConsensus
