from __future__ import annotations

from typing import Any

from ..schemas.requests import ClinicalData


def build_feature_vector(data: ClinicalData) -> list[float]:
    return [
        float(data.pregnancies),
        float(data.glucose),
        float(data.blood_pressure),
        float(data.skin_thickness),
        float(data.insulin),
        float(data.bmi),
        float(data.diabetes_pedigree_function),
        float(data.age),
    ]


def engineer_features(data: ClinicalData) -> dict[str, float | str]:
    # Clip ratio to handle insulin=0 (common missing-data placeholder in Pima dataset).
    # Without clipping, ratio=148 when insulin=0 causes extreme z-score and collapses model output.
    glucose_insulin_index = min(data.glucose / (data.insulin + 1.0), 10.0)
    bmi_category = "normal"
    if data.bmi >= 30:
        bmi_category = "obese"
    elif data.bmi >= 25:
        bmi_category = "overweight"

    if data.age < 35:
        age_group = "young"
    elif data.age < 55:
        age_group = "adult"
    else:
        age_group = "senior"

    glucose_bmi_index = (data.glucose / 100.0) * (data.bmi / 30.0)
    age_risk_index = data.age / 50.0
    family_risk_index = data.diabetes_pedigree_function * 1.5

    return {
        "glucose_insulin_index": round(glucose_insulin_index, 4),
        "glucose_bmi_index": round(glucose_bmi_index, 4),
        "age_risk_index": round(age_risk_index, 4),
        "family_risk_index": round(family_risk_index, 4),
        "bmi_category": bmi_category,
        "age_group": age_group,
    }


def risk_level_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "moderate"
    return "low"


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(score, 0.0) for score in scores.values())
    if total <= 0:
        count = max(len(scores), 1)
        return {label: 1.0 / count for label in scores}
    return {label: max(score, 0.0) / total for label, score in scores.items()}


def symptoms_match_diabetes(text: str, top_label: str) -> bool:
    lowered = text.lower()
    diabetes_keywords = ["thirst", "urinate", "urination", "polyuria", "fatigue", "tired", "blurred", "vision", "weight loss"]
    if top_label.lower().startswith("diabetes"):
        return any(keyword in lowered for keyword in diabetes_keywords)
    return False


def build_llm_context(clinical_data: ClinicalData, risk_probability: float, symptoms_text: str) -> dict[str, Any]:
    return {
        "clinical_data": {
            "pregnancies": clinical_data.pregnancies,
            "glucose": clinical_data.glucose,
            "blood_pressure": clinical_data.blood_pressure,
            "skin_thickness": clinical_data.skin_thickness,
            "insulin": clinical_data.insulin,
            "bmi": clinical_data.bmi,
            "diabetes_pedigree_function": clinical_data.diabetes_pedigree_function,
            "age": clinical_data.age,
        },
        "risk_probability": round(risk_probability, 4),
        "risk_level": risk_level_from_probability(risk_probability),
        "symptoms_text": symptoms_text,
    }
