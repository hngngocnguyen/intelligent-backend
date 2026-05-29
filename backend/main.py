from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models.classic_model import get_classic_service
from .models.hf_model import get_huggingface_service
from .models.llm_model import get_llm_service
from .routers.classic import router as classic_router
from .routers.huggingface import router as huggingface_router
from .routers.llm import router as llm_router
from .routers.pipeline import router as pipeline_router
from .schemas.requests import HealthResponse, ModelInfoResponse
from .utils.logger import configure_logging


configure_logging()

app = FastAPI(
    title="Backend Intelligent - Diabetes Risk",
    description=(
        "Prototype FastAPI backend combining a tabular deep-learning model, "
        "a Hugging Face zero-shot analyzer, and an "
        "OpenRouter-powered report generator."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classic_router)
app.include_router(huggingface_router)
app.include_router(llm_router)
app.include_router(pipeline_router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "service": "backend-intelligent",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    classic_service = get_classic_service()
    hf_service = get_huggingface_service()
    llm_service = get_llm_service()
    return HealthResponse(
        status="ok",
        models={
            "classic": classic_service.health_snapshot(),
            "huggingface": hf_service.health_snapshot(),
            "llm": llm_service.health_snapshot(),
        },
    )


@app.get("/models/info", response_model=ModelInfoResponse, tags=["system"])
def models_info() -> ModelInfoResponse:
    classic_service = get_classic_service()
    hf_service = get_huggingface_service()
    llm_service = get_llm_service()
    return ModelInfoResponse(
        classic_model=classic_service.model_info(),
        huggingface_model=hf_service.model_info(),
        llm_model=llm_service.model_info(),
    )
