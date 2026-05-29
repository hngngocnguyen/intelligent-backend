from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..schemas.requests import ClinicalData, ClassicPredictionResponse
from ..utils.preprocessing import (
    build_feature_vector,
    engineer_features,
    risk_level_from_probability,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ClassicModelArtifact:
    variant: str
    path: Path
    loaded: bool


class ClassicDiabetesService:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        self.model_dir = model_dir or base_dir / "saved_models"
        self._tf = None
        self._scalers: dict[str, Any] = {}  # per-variant scalers
        self._model_cache: dict[str, Any] = {}  # loaded Keras models (loaded once)
        self._artifacts: dict[str, ClassicModelArtifact] = {}
        self._load_available_artifacts()

    def _try_import_tensorflow(self) -> Any:
        if self._tf is not None:
            return self._tf
        try:
            import tensorflow as tf  # type: ignore

            self._tf = tf
        except Exception:  # pragma: no cover - optional dependency
            self._tf = False
        return self._tf

    def _load_available_artifacts(self) -> None:
        for variant in ("mlp_v1", "mlp_v2", "mlp_v3"):
            # look for model in model_dir, or in parent (project-level) saved_models
            model_path = self.model_dir / f"{variant}.keras"
            if not model_path.exists():
                alt = self.model_dir.parent / f"{variant}.keras"
                if alt.exists():
                    model_path = alt

            self._artifacts[variant] = ClassicModelArtifact(
                variant=variant,
                path=model_path,
                loaded=model_path.exists(),
            )

        # Load per-variant scalers.
        # Priority:
        #   1. scaler_v{N}.pkl  (from the HPO script, in model_dir or parent)
        #   2. scaler.pkl       (12-feature scaler, used for mlp_v3)
        #   3. scaler_base.pkl  (8-feature scaler, used for mlp_v1 and mlp_v2)
        try:
            import joblib

            # Helper: find a file in model_dir, then in parent
            def _find(name: str) -> Path | None:
                p = self.model_dir / name
                if p.exists():
                    return p
                alt = self.model_dir.parent / name
                return alt if alt.exists() else None

            # Try dedicated per-variant scaler files first (from HPO script)
            for suffix, variant_key in (("v1", "mlp_v1"), ("v2", "mlp_v2"), ("v3", "mlp_v3")):
                path = _find(f"scaler_{suffix}.pkl")
                if path:
                    try:
                        self._scalers[variant_key] = joblib.load(str(path))
                        logger.info("Loaded scaler for %s from %s", variant_key, path)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Unable to load scaler for %s: %s", variant_key, exc)

            # Fall back to scaler.pkl (12 features) for mlp_v3 if not yet loaded
            if "mlp_v3" not in self._scalers:
                path = _find("scaler.pkl")
                if path:
                    try:
                        self._scalers["mlp_v3"] = joblib.load(str(path))
                        logger.info("Loaded 12-feat scaler for mlp_v3 from %s", path)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Unable to load scaler.pkl: %s", exc)

            # Fall back to scaler_base.pkl (8 features) for mlp_v1 / mlp_v2 if not yet loaded
            base_path = _find("scaler_base.pkl")
            if base_path:
                try:
                    base_scaler = joblib.load(str(base_path))
                    for variant_key in ("mlp_v1", "mlp_v2"):
                        if variant_key not in self._scalers:
                            self._scalers[variant_key] = base_scaler
                            logger.info("Loaded 8-feat scaler for %s from %s", variant_key, base_path)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Unable to load scaler_base.pkl: %s", exc)

        except ImportError:
            logger.warning("joblib not available — scalers will not be loaded")
        # Load optional serving configuration (thresholds, chosen variant)
        try:
            import json

            cfg_path = self.model_dir / "serving_config.json"
            if not cfg_path.exists():
                alt_cfg = self.model_dir.parent / "serving_config.json"
                if alt_cfg.exists():
                    cfg_path = alt_cfg

            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                    self._serving_config = cfg
            else:
                self._serving_config = {}
        except Exception:
            self._serving_config = {}

    def model_info(self) -> dict[str, Any]:
        loaded_variants = [
            name
            for name, artifact in self._artifacts.items()
            if artifact.loaded
        ]
        return {
            "service": "classic",
            "available_variants": list(self._artifacts),
            "loaded_variants": loaded_variants,
            "uses_tensorflow": bool(self._try_import_tensorflow()),
            "scalers_loaded": list(self._scalers.keys()),
            "fallback_mode": not any(
                artifact.loaded for artifact in self._artifacts.values()
            ),
        }

    def health_snapshot(self) -> dict[str, Any]:
        loaded_variants = [
            name
            for name, artifact in self._artifacts.items()
            if artifact.loaded
        ]
        return {
            "ready": True,
            "fallback_mode": not any(
                artifact.loaded for artifact in self._artifacts.values()
            ),
            "loaded_variants": loaded_variants,
            "scalers_loaded": list(self._scalers.keys()),
        }

    def predict(
        self,
        clinical_data: ClinicalData,
        model_version: str = "mlp_v3",
    ) -> ClassicPredictionResponse:
        artifact = self._artifacts.get(model_version)
        if artifact is None:
            artifact = self._artifacts["mlp_v3"]

        probability = self._predict_probability(
            clinical_data,
            artifact.variant,
        )
        # Determine threshold: check serving config, then default to 0.5
        threshold = 0.5
        try:
            thr = self._serving_config.get("thresholds", {}).get(artifact.variant)
            if thr is not None:
                threshold = float(thr)
        except Exception:
            threshold = 0.5

        prediction = "diabetic" if probability >= threshold else "non-diabetic"

        return ClassicPredictionResponse(
            model_version=artifact.variant,
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level_from_probability(probability),
            inference_time_ms=self._estimate_inference_time(
                artifact.variant,
                artifact.loaded,
            ),
            feature_summary=engineer_features(clinical_data),
        )

    def _predict_probability(
        self,
        clinical_data: ClinicalData,
        variant: str,
    ) -> float:
        if self._can_use_artifact(variant):
            return self._predict_with_tensorflow(clinical_data, variant)
        return self._heuristic_probability(clinical_data, variant)

    def _can_use_artifact(self, variant: str) -> bool:
        artifact = self._artifacts.get(variant)
        return bool(
            artifact
            and artifact.loaded
            and self._try_import_tensorflow()
        )

    def _predict_with_tensorflow(
        self,
        clinical_data: ClinicalData,
        variant: str,
    ) -> float:
        tf = self._try_import_tensorflow()
        if not tf:
            return self._heuristic_probability(clinical_data, variant)

        artifact = self._artifacts[variant]
        if variant not in self._model_cache:
            self._model_cache[variant] = tf.keras.models.load_model(str(artifact.path))
        model = self._model_cache[variant]

        features_8 = build_feature_vector(clinical_data)
        engineered = engineer_features(clinical_data)
        
        # In NB01, the 4 engineered features were added BEFORE the scaler.
        # However, the dataset columns might be in a specific order:
        # We append them to the base features.
        features_12 = features_8 + [
            float(engineered["glucose_insulin_index"]),
            float(engineered["glucose_bmi_index"]),
            float(engineered["age_risk_index"]),
            float(engineered["family_risk_index"]),
        ]

        scaler = self._scalers.get(variant)
        if scaler is not None:
            try:
                # Try scaling 12 features first (v3 was trained on 12 features)
                scaled = scaler.transform([features_12])[0].tolist()
                final_features = scaled if variant == "mlp_v3" else scaled[:8]
            except ValueError:
                # Fallback: scaler was fit on 8 base features
                scaled_8 = scaler.transform([features_8])[0].tolist()
                final_features = scaled_8 + features_12[8:] if variant == "mlp_v3" else scaled_8
        else:
            final_features = features_12 if variant == "mlp_v3" else features_8

        prediction = model.predict(tf.constant([final_features]), verbose=0)
        return float(prediction.reshape(-1)[0])

    def _heuristic_probability(
        self,
        clinical_data: ClinicalData,
        variant: str,
    ) -> float:
        glucose = clinical_data.glucose
        bmi = clinical_data.bmi
        age = clinical_data.age
        pedigree = clinical_data.diabetes_pedigree_function
        insulin = clinical_data.insulin
        blood_pressure = clinical_data.blood_pressure
        pregnancies = clinical_data.pregnancies
        skin_thickness = clinical_data.skin_thickness

        base_logit = (
            -7.25
            + 0.045 * glucose
            + 0.06 * bmi
            + 0.03 * age
            + 0.85 * pedigree
            + 0.012 * insulin
            + 0.018 * blood_pressure
            + 0.08 * pregnancies
            + 0.018 * skin_thickness
        )

        if variant == "mlp_v1":
            base_logit -= 0.5
        elif variant == "mlp_v2":
            base_logit -= 0.1
        else:
            features = engineer_features(clinical_data)
            base_logit += 0.35 * float(features["glucose_bmi_index"])
            base_logit += 0.18 * float(features["age_risk_index"])
            base_logit += 0.25 * float(features["family_risk_index"])

        probability = 1.0 / (1.0 + math.exp(-base_logit))

        if variant == "mlp_v2":
            probability = 0.9 * probability + 0.05
        elif variant == "mlp_v3":
            probability = 0.85 * probability + 0.08

        return max(0.02, min(0.98, probability))

    @staticmethod
    def _estimate_inference_time(variant: str, loaded: bool) -> int:
        if loaded:
            return {"mlp_v1": 7, "mlp_v2": 10, "mlp_v3": 12}.get(variant, 10)
        return {"mlp_v1": 2, "mlp_v2": 3, "mlp_v3": 4}.get(variant, 3)


_CLASSIC_SERVICE: ClassicDiabetesService | None = None


def get_classic_service() -> ClassicDiabetesService:
    global _CLASSIC_SERVICE
    if _CLASSIC_SERVICE is None:
        _CLASSIC_SERVICE = ClassicDiabetesService()
    return _CLASSIC_SERVICE
