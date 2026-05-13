from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.crud.model_loader import get_model
from backend.routers.analyze import router as analyze_router
from backend.routers.predict import router as predict_router

load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="Audio–visual consistency API",
    description="Runs AudioVisualFusion (R3D-18 + ResNet-18) on uploaded clips.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(analyze_router)


@app.get("/health")
def health():
    try:
        model, _ = get_model()
        total_params = sum(p.numel() for p in model.parameters())
        return {"status": "ok", "total_parameters": total_params}
    except Exception as e:
        return {"status": "ok", "total_parameters": None, "error": str(e)}
