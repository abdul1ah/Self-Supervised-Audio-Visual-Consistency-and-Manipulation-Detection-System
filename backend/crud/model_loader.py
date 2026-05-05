from __future__ import annotations

import sys
from pathlib import Path

import torch

from backend.config import CHECKPOINT_PATH, SRC_DIR

_model_cache: torch.nn.Module | None = None
_device_cache: torch.device | None = None


def _import_audio_visual_fusion():
    # src/models.py only depends on torch; avoid importing src/config (Kaggle paths).
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from models import AudioVisualFusion  # noqa: WPS433 — runtime path hook

    return AudioVisualFusion


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Train with src/train.py or set CHECKPOINT_PATH."
        )
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(str(k).startswith("module.") for k in state.keys()):
        state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def get_model(device: torch.device | None = None) -> tuple[torch.nn.Module, torch.device]:
    global _model_cache, _device_cache
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _model_cache is not None and _device_cache == device:
        return _model_cache, device

    AudioVisualFusion = _import_audio_visual_fusion()
    model = AudioVisualFusion(pretrained=False)
    load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()
    _model_cache = model
    _device_cache = device
    return model, device
