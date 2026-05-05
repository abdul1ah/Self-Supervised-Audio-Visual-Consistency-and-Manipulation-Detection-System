"""Runtime configuration for the FastAPI service (paths are local, not Kaggle)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"

CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(PROJECT_ROOT / "backend" / "uploads")))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

MIN_FRAMES = 45
TARGET_FRAME_SIZE = (224, 224)
NUM_MEL_BINS = 128
AUDIO_SAMPLE_RATE = 48000
