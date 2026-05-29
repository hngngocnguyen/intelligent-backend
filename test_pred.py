import sys
from pathlib import Path
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(r"d:\Education\M2 - IA - NEXA\S2\05. Introduction aux modèles IA génératives - TRONC\intelligent-backend").resolve()))

from backend.schemas.requests import ClinicalData
from backend.models.classic_model import get_classic_service

try:
    svc = get_classic_service()
    print("Models:", svc.model_info())
    data = ClinicalData(pregnancies=2, glucose=120, blood_pressure=72, skin_thickness=25, insulin=90, bmi=32.0, diabetes_pedigree_function=0.47, age=45)
    res = svc.predict(data, "mlp_v1")
    print("Result v1:", res)
    res2 = svc.predict(data, "mlp_v3")
    print("Result v3:", res2)
except Exception as e:
    import traceback
    traceback.print_exc()
