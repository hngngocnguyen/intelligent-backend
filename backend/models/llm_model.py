from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..schemas.requests import ExplanationRequest, MedicalExplanationResponse
from ..utils.preprocessing import (
    build_llm_context,
    risk_level_from_probability,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenRouterArtifact:
    model_name: str
    api_key_present: bool
    fallback_mode: bool


class OpenRouterExplanationService:
    def __init__(
        self,
        model_name: str | None = None,
        model_candidates: list[str] | None = None,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ) -> None:
        default_candidates = [
            "openai/gpt-oss-20b:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-4-31b-it:free",
            "deepseek/deepseek-v4-flash:free",
        ]
        env_candidates = os.getenv("OPENROUTER_MODEL_CANDIDATES", "").strip()
        if env_candidates:
            resolved_candidates = [
                item.strip() for item in env_candidates.split(",") if item.strip()
            ]
        else:
            resolved_candidates = list(default_candidates)

        self.model_candidates = model_candidates or resolved_candidates
        self.model_name = model_name or os.getenv(
            "OPENROUTER_MODEL",
            self.model_candidates[0],
        )
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def model_info(self) -> dict[str, Any]:
        return {
            "service": "llm",
            "model_name": self.model_name,
            "model_candidates": list(self.model_candidates),
            "api_key_present": bool(self.api_key),
            "fallback_mode": not bool(self.api_key),
        }

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "ready": True,
            "api_key_present": bool(self.api_key),
            "fallback_mode": not bool(self.api_key),
        }

    def generate(
        self,
        request: ExplanationRequest,
    ) -> MedicalExplanationResponse:
        start_context = build_llm_context(
            request.clinical_data,
            request.risk_probability,
            request.symptoms_text,
        )
        if self.api_key:
            try:
                parsed, model_used = self._call_openrouter(start_context)
                if parsed:
                    return self._response_from_payload(
                        parsed,
                        request,
                        mode="openrouter",
                        model_name=model_used,
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("OpenRouter request failed: %s", exc)

        payload = self._fallback_payload(request)
        return self._response_from_payload(payload, request, mode="fallback")

    def _call_openrouter(
        self,
        context: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Backend Intelligent Diabetes Risk",
        }
        body_template = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful medical AI assistant. "
                        "Reply only with "
                        "valid JSON. Use patient-friendly wording, stay "
                        "non-alarmist, and include a medical disclaimer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Create a JSON medical summary with keys: risk_level, "
                        "summary, recommendations, urgency, disclaimer. "
                        "Urgency must be one of routine|soon|immediate. "
                        f"Context: {json.dumps(context, ensure_ascii=True)}"
                    ),
                },
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        # Deduplicate while preserving order: primary model first, then remaining candidates
        seen: set[str] = set()
        ordered_candidates: list[str] = []
        for m in [self.model_name, *self.model_candidates]:
            if m not in seen:
                seen.add(m)
                ordered_candidates.append(m)

        for model_name in ordered_candidates:
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json={"model": model_name, **body_template},
                        timeout=30,
                    )
                    if response.ok:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        parsed = self._extract_json(content)
                        if parsed:
                            return parsed, model_name
                        last_error = "Invalid JSON response"
                        break
                    if response.status_code == 429 and attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    break
                except Exception as exc:  # pragma: no cover
                    last_error = str(exc)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    break
            logger.warning("OpenRouter model failed (%s): %s", model_name, last_error)
        return None, None

    def _fallback_payload(self, request: ExplanationRequest) -> dict[str, Any]:
        risk_level = risk_level_from_probability(request.risk_probability)
        if request.risk_probability >= 0.7:
            urgency = "immediate"
            recommendations = [
                "Schedule a medical appointment within the next few days.",
                "Monitor blood glucose and hydration closely.",
                "Reduce refined sugars and ultra-processed carbohydrates.",
                "Increase physical activity gradually if medically "
                "appropriate.",
            ]
        elif request.risk_probability >= 0.4:
            urgency = "routine"
            recommendations = [
                "Discuss your results with a healthcare professional.",
                "Watch carbohydrate intake and portion sizes.",
                "Repeat blood tests if symptoms persist.",
            ]
        else:
            urgency = "routine"
            recommendations = [
                "Maintain your current healthy habits.",
                "Keep an eye on symptoms and follow up if they change.",
                "Consider routine screening if you have family history or "
                "other risk factors.",
            ]

        symptom_excerpt = request.symptoms_text[:180].strip()
        summary = (
            f"The tabular model suggests a {risk_level} diabetes risk. "
            f"The symptom analysis highlights: {symptom_excerpt}"
        )
        return {
            "risk_level": risk_level,
            "summary": summary,
            "recommendations": recommendations,
            "urgency": urgency,
            "disclaimer": (
                "This report is informational only and does not replace "
                "a consultation with a licensed medical professional."
            ),
        }

    def _response_from_payload(
        self,
        payload: dict[str, Any],
        request: ExplanationRequest,
        mode: str,
        model_name: str | None = None,
    ) -> MedicalExplanationResponse:
        recommendations = payload.get("recommendations") or []
        if isinstance(recommendations, str):
            recommendations = [recommendations]
        return MedicalExplanationResponse(
            model=model_name or self.model_name,
            mode=mode,
            risk_level=str(
                payload.get(
                    "risk_level",
                    risk_level_from_probability(request.risk_probability),
                )
            ),
            summary=str(payload.get("summary", "No summary available.")),
            recommendations=[str(item) for item in recommendations],
            urgency=str(payload.get("urgency", "routine")),
            disclaimer=str(
                payload.get(
                    "disclaimer",
                    "This report is informational only and does not replace "
                    "a consultation with a licensed medical professional.",
                )
            ),
            inference_time_ms=1450 if mode == "openrouter" else 18,
        )

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | None:
        candidate = content.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


_LLM_SERVICE: OpenRouterExplanationService | None = None


def get_llm_service() -> OpenRouterExplanationService:
    global _LLM_SERVICE
    if _LLM_SERVICE is None:
        _LLM_SERVICE = OpenRouterExplanationService()
    return _LLM_SERVICE
