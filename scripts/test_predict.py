import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.models.classic_model import get_classic_service
from backend.schemas.requests import ClinicalData

svc = get_classic_service()
print('Model info:', svc.model_info())

sample = ClinicalData(
    pregnancies=2,
    glucose=150.0,
    blood_pressure=80.0,
    skin_thickness=20.0,
    insulin=85.0,
    bmi=32.0,
    diabetes_pedigree_function=0.5,
    age=45,
)
res = svc.predict(sample, model_version='mlp_v1')
print('Prediction:', res.json())
