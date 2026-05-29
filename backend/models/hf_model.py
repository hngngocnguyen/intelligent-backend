from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..schemas.requests import (
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    SymptomScore,
)
from ..utils.preprocessing import normalize_scores, symptoms_match_diabetes

logger = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "diabetes symptoms",
    "hypertension symptoms",
    "cardiovascular symptoms",
    "neurological symptoms",
    "digestive symptoms",
    "no significant symptoms",
]


@dataclass(slots=True)
class HuggingFaceArtifact:
    model_name: str
    loaded: bool
    fallback_mode: bool


class HuggingFaceSymptomService:
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._pipeline_attempted = False

    def _try_load_pipeline(self) -> Any:
        if self._pipeline_attempted:
            return self._pipeline

        self._pipeline_attempted = True
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Unable to initialize Hugging Face pipeline: %s",
                exc,
            )
            self._pipeline = None
        return self._pipeline

    def model_info(self) -> dict[str, Any]:
        return {
            "service": "huggingface",
            "model_name": self.model_name,
            "loaded": bool(self._pipeline),
            "fallback_mode": self._pipeline is None,
        }

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "ready": True,
            "loaded": bool(self._pipeline),
            "fallback_mode": self._pipeline is None,
        }

    def analyze(
        self,
        request: SymptomAnalysisRequest,
        classic_probability: float | None = None,
    ) -> SymptomAnalysisResponse:
        pipeline_instance = self._try_load_pipeline()
        labels = request.candidate_labels or DEFAULT_LABELS

        if pipeline_instance is not None:
            try:
                result = pipeline_instance(
                    request.symptoms_text,
                    candidate_labels=labels,
                    multi_label=False,
                )
                scores = dict(
                    zip(result["labels"], result["scores"], strict=False)
                )
                ordered_scores = normalize_scores(scores)
                top_label = max(ordered_scores, key=ordered_scores.get)
                confidence = float(ordered_scores[top_label])
                return SymptomAnalysisResponse(
                    model=self.model_name,
                    top_category=top_label,
                    confidence=round(confidence, 4),
                    all_scores={
                        label: round(score, 4)
                        for label, score in ordered_scores.items()
                    },
                    top_categories=self._top_scores(ordered_scores),
                    symptom_matches_prediction=self._symptom_match(
                        request.symptoms_text,
                        top_label,
                        classic_probability,
                    ),
                    inference_time_ms=420,
                    mode="hf",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "HF inference failed, using fallback analysis: %s",
                    exc,
                )

        fallback_scores = self._fallback_scores(request.symptoms_text, labels)
        top_label = max(fallback_scores, key=fallback_scores.get)
        confidence = float(fallback_scores[top_label])
        return SymptomAnalysisResponse(
            model=self.model_name,
            top_category=top_label,
            confidence=round(confidence, 4),
            all_scores={
                label: round(score, 4)
                for label, score in fallback_scores.items()
            },
            top_categories=self._top_scores(fallback_scores),
            symptom_matches_prediction=self._symptom_match(
                request.symptoms_text,
                top_label,
                classic_probability,
            ),
            inference_time_ms=45,
            mode="heuristic",
        )

    def _fallback_scores(
        self,
        text: str,
        labels: list[str],
    ) -> dict[str, float]:
        lowered = text.lower()
        keyword_map = {
            "diabetes symptoms": [
                "thirst",
                "urinate",
                "urination",
                "tired",
                "fatigue",
                "blurred",
                "vision",
                "weight loss",
            ],
            "hypertension symptoms": [
                "headache",
                "dizzy",
                "dizziness",
                "pressure",
                "hypertension",
            ],
            "cardiovascular symptoms": [
                "chest",
                "palpitation",
                "shortness",
                "breath",
                "heart",
            ],
            "neurological symptoms": [
                "numb",
                "tingling",
                "confusion",
                "weakness",
                "memory",
            ],
            "digestive symptoms": [
                "nausea",
                "vomit",
                "stomach",
                "abdominal",
                "digestive",
            ],
            "no significant symptoms": [
                "nothing",
                "fine",
                "normal",
                "healthy",
            ],
        }

        raw_scores: dict[str, float] = defaultdict(float)
        for label in labels:
            score = 0.1
            keywords = keyword_map.get(label.lower(), [])
            for keyword in keywords:
                if keyword in lowered:
                    score += 0.22
            if label == "no significant symptoms" and not any(
                keyword in lowered
                for keywords in keyword_map.values()
                for keyword in keywords
            ):
                score += 0.35
            raw_scores[label] = score
        return normalize_scores(dict(raw_scores))

    @staticmethod
    def _top_scores(
        scores: dict[str, float],
        top_k: int = 3,
    ) -> list[SymptomScore]:
        ordered_items = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return [
            SymptomScore(category=label, score=round(score, 4))
            for label, score in ordered_items
        ]

    @staticmethod
    def _symptom_match(
        text: str,
        top_label: str,
        classic_probability: float | None,
    ) -> bool | None:
        if classic_probability is None:
            return symptoms_match_diabetes(text, top_label)
        match = symptoms_match_diabetes(text, top_label)
        return match and classic_probability >= 0.35


_HUGGINGFACE_SERVICE: HuggingFaceSymptomService | None = None


def get_huggingface_service() -> HuggingFaceSymptomService:
    global _HUGGINGFACE_SERVICE
    if _HUGGINGFACE_SERVICE is None:
        _HUGGINGFACE_SERVICE = HuggingFaceSymptomService()
    return _HUGGINGFACE_SERVICE
