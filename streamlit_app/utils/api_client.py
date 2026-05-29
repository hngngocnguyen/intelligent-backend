from __future__ import annotations

import os
from typing import Any

import requests


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        default_url = "http://localhost:8000"
        configured_url = base_url or os.getenv("BACKEND_URL", default_url)
        self.base_url = configured_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

    def safe_health(self) -> dict[str, Any]:
        try:
            return self._request("GET", "/health")
        except Exception:
            return {"status": "offline"}

    def safe_models_info(self) -> dict[str, Any]:
        try:
            return self._request("GET", "/models/info")
        except Exception:
            return {}

    def predict_tabular(
        self,
        clinical_data: dict[str, Any],
        model_version: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/predict/tabular?model_version={model_version}",
            clinical_data,
        )

    def analyze_symptoms(self, symptoms_text: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/analyze/symptoms",
            {"symptoms_text": symptoms_text},
        )

    def explain(
        self,
        clinical_data: dict[str, Any],
        symptoms_text: str,
        risk_probability: float,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/explain",
            {
                "clinical_data": clinical_data,
                "symptoms_text": symptoms_text,
                "risk_probability": risk_probability,
            },
        )

    def full_pipeline(
        self,
        clinical_data: dict[str, Any],
        symptoms_text: str,
        model_version: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "clinical_data": clinical_data,
            "symptoms_text": symptoms_text,
        }
        if model_version:
            payload["model_version"] = model_version
        return self._request(
            "POST",
            "/full-pipeline",
            payload,
        )
