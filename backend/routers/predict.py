from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import MAX_UPLOAD_MB
from backend.crud.inference_crud import run_av_prediction, save_upload
from backend.crud.video_kind import VideoKind

router = APIRouter(prefix="/api", tags=["prediction"])

_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _suffix(filename: str | None) -> str:
    if not filename:
        return ".mp4"
    p = Path(filename).suffix.lower()
    return p if p in {".mp4", ".webm", ".mov", ".mkv", ".avi"} else ".mp4"


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    video_kind: str = Form(
        ...,
        description='Either "youtube" (natural background; face-centric crop) or '
        '"dataset" (clean / full-frame resize, RAVDESS-style).',
    ),
):
    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    try:
        kind = VideoKind(video_kind)
    except ValueError:
        raise HTTPException(
            400,
            'video_kind must be "youtube" or "dataset".',
        )

    path = await save_upload(raw, _suffix(file.filename))
    try:
        result = await run_av_prediction(path, kind)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}") from e

    return result
