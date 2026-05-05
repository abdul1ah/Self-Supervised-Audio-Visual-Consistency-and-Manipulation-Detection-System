from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from backend.config import UPLOAD_DIR
from backend.crud.video_kind import VideoKind


def _interpret(probability: float, threshold: float = 0.5) -> str:
    if probability >= threshold:
        return (
            "High audio–visual consistency: the soundtrack appears aligned with the "
            f"visuals for this clip (score {probability:.3f})."
        )
    return (
        "Low audio–visual consistency: the clip may contain a mismatch or manipulation, "
        f"or the domain differs from training (score {probability:.3f})."
    )


async def save_upload(contents: bytes, suffix: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / name
    path.write_bytes(contents)
    return path


def predict_from_path(video_path: Path, kind: VideoKind) -> dict:
    import torch

    from backend.crud.model_loader import get_model
    from backend.crud.preprocessing import load_video_tensors

    model, device = get_model()
    video, audio = load_video_tensors(video_path, kind, device)
    video = video.to(device)
    audio = audio.to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(video, audio)
        else:
            logits = model(video, audio)

    probability = float(torch.sigmoid(logits).squeeze().cpu().item())
    return {
        "match_probability": probability,
        "interpretation": _interpret(probability),
        "video_kind": kind.value,
    }


async def run_av_prediction(saved_path: Path, kind: VideoKind) -> dict:
    try:
        return await asyncio.to_thread(predict_from_path, saved_path, kind)
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            pass
