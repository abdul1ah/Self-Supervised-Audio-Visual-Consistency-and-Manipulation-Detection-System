"""
Video/audio preprocessing aligned with src/preprocess.py and src/dataset.py (eval path).
- dataset / clean-background: full-frame resize (training distribution for RAVDESS-style clips).
- youtube / natural background: optional MTCNN face crop on a anchor frame, then per-frame crop + resize.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from PIL import Image

from backend.config import (
    AUDIO_SAMPLE_RATE,
    MIN_FRAMES,
    NUM_MEL_BINS,
    TARGET_FRAME_SIZE,
)
from backend.crud.video_kind import VideoKind


def _to_device_float_audio(a: torch.Tensor) -> torch.Tensor:
    if a.dtype in (torch.int16, torch.int32):
        a = a.float() / float(torch.iinfo(a.dtype).max)
    else:
        a = a.float()
    return a


def _torchvision_read_video(path: Path):
    """Use torchvision when this build includes decoders (often unavailable on Windows wheels)."""
    io_mod = getattr(torchvision, "io", None)
    read_fn = getattr(io_mod, "read_video", None) if io_mod is not None else None
    if read_fn is None or not callable(read_fn):
        return None
    try:
        return read_fn(str(path), pts_unit="sec", output_format="TCHW")
    except Exception:
        return None


def _decode_full_audio_pyav(path: Path) -> Tuple[torch.Tensor, int]:
    """Return float audio (C, num_samples) and sample rate."""
    import av

    container = av.open(str(path))
    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return torch.zeros(1, 0), AUDIO_SAMPLE_RATE
        astream = audio_streams[0]
        astream.thread_type = "AUTO"
        sr = int(astream.sample_rate or AUDIO_SAMPLE_RATE)
        chunks: list[torch.Tensor] = []
        for packet in container.demux(astream):
            for frame in packet.decode():
                arr = frame.to_ndarray()
                if arr.dtype == np.int16:
                    tf = torch.from_numpy(arr.astype(np.float32) / 32768.0)
                elif arr.dtype == np.int32:
                    tf = torch.from_numpy(arr.astype(np.float32) / 2147483648.0)
                else:
                    tf = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                if tf.ndim == 1:
                    tf = tf.unsqueeze(0)
                chunks.append(tf)
        if not chunks:
            return torch.zeros(1, 0), sr
        return torch.cat(chunks, dim=1), sr
    finally:
        container.close()


def _read_video_opencv_audio_pyav(path: Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Read center 45-frame clip (TCHW uint8) + matching audio slice (C, samples),
    matching src/preprocess.py timing.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nframes < MIN_FRAMES:
            raise ValueError(f"Video must have at least {MIN_FRAMES} frames; got {nframes}")
        start_v = (nframes - MIN_FRAMES) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_v))

        frames: list[torch.Tensor] = []
        for _ in range(MIN_FRAMES):
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb).clone().permute(2, 0, 1))
        if len(frames) < MIN_FRAMES:
            raise ValueError(
                f"Expected {MIN_FRAMES} frames from video segment; got {len(frames)}."
            )
        vframes = torch.stack(frames, dim=0).to(torch.uint8)
    finally:
        cap.release()

    a_full, audio_sr = _decode_full_audio_pyav(path)
    if a_full.numel() == 0:
        raise ValueError("No audio stream found; the model needs paired audio and video.")

    start_sec = start_v / fps
    end_sec = (start_v + MIN_FRAMES) / fps
    a_start = int(start_sec * audio_sr)
    a_end = int(end_sec * audio_sr)
    a_end = max(a_end, a_start + 1)
    a_end = min(a_end, a_full.shape[1])
    a_clip = a_full[:, a_start:a_end]
    if a_clip.shape[1] < 1:
        raise ValueError("Audio track is too short for the selected video segment.")

    info = {"video_fps": fps, "audio_fps": float(audio_sr)}
    return vframes, a_clip, info


def _read_video_unified(path: Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    tv = _torchvision_read_video(path)
    if tv is not None:
        vframes, aframes, info = tv
        if aframes.numel() == 0:
            raise ValueError("No audio stream found; the model needs paired audio and video.")
        video_fps = float(info.get("video_fps", 30.0) or 30.0)
        audio_fps = float(info.get("audio_fps", AUDIO_SAMPLE_RATE) or AUDIO_SAMPLE_RATE)
        v_clip, a_clip = _extract_center_clip(vframes, aframes, video_fps, audio_fps)
        return v_clip, a_clip, {"video_fps": video_fps, "audio_fps": audio_fps}

    return _read_video_opencv_audio_pyav(path)


def _extract_center_clip(
    vframes: torch.Tensor,
    aframes: torch.Tensor,
    video_fps: float,
    audio_fps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return 45-frame TCHW uint8 clip and mono audio slice matching preprocess.py."""
    total_v = vframes.shape[0]
    if total_v < MIN_FRAMES:
        raise ValueError(f"Video must have at least {MIN_FRAMES} frames; got {total_v}")

    start_v = (total_v - MIN_FRAMES) // 2
    v_clip = vframes[start_v : start_v + MIN_FRAMES]

    v_fps = float(video_fps) if video_fps else 30.0
    a_fps = float(audio_fps) if audio_fps else float(AUDIO_SAMPLE_RATE)

    a_start = int((start_v / v_fps) * a_fps)
    a_end = int(((start_v + MIN_FRAMES) / v_fps) * a_fps)
    a_clip = aframes[:, a_start:a_end]

    if a_clip.shape[1] < 1:
        raise ValueError("Audio track is too short for the selected video segment.")

    if a_clip.shape[0] > 1:
        a_clip = torch.mean(a_clip, dim=0, keepdim=True)

    return v_clip, a_clip


def _expand_box(
    box: np.ndarray,
    width: int,
    height: int,
    expand: float = 0.35,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w2, h2 = w * (1.0 + expand), h * (1.0 + expand)
    nx1 = int(max(0, cx - w2 / 2))
    ny1 = int(max(0, cy - h2 / 2))
    nx2 = int(min(width, cx + w2 / 2))
    ny2 = int(min(height, cy + h2 / 2))
    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        return 0, 0, width, height
    return nx1, ny1, nx2, ny2


def _face_box_from_frame(
    frame_chw: torch.Tensor,
    mtcnn,
) -> Optional[Tuple[int, int, int, int]]:
    """frame_chw uint8 or float CHW."""
    c, h, w = frame_chw.shape
    np_img = frame_chw.detach().cpu().numpy()
    if np_img.dtype != np.uint8:
        np_img = (np_img * 255.0).clip(0, 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1, 2, 0))
    if np_img.shape[2] == 1:
        np_img = np.repeat(np_img, 3, axis=2)
    pil = Image.fromarray(np_img)
    boxes, _ = mtcnn.detect(pil)
    if boxes is None or len(boxes) == 0:
        return None
    return _expand_box(boxes[0], w, h)


def _spatial_resize_clip_tchw(v_clip: torch.Tensor) -> torch.Tensor:
    resize = torchvision.transforms.Resize(TARGET_FRAME_SIZE)
    return resize(v_clip)


def _face_crop_clip_tchw(v_clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        return _spatial_resize_clip_tchw(v_clip)

    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    mid = v_clip.shape[0] // 2
    anchor = v_clip[mid]
    box = _face_box_from_frame(anchor, mtcnn)
    if box is None:
        return _spatial_resize_clip_tchw(v_clip)
    x1, y1, x2, y2 = box
    out = []
    for t in range(v_clip.shape[0]):
        frame = v_clip[t]
        cropped = frame[:, y1:y2, x1:x2]
        if cropped.shape[1] < 2 or cropped.shape[2] < 2:
            cropped = frame
        cropped = torchvision.transforms.functional.resize(cropped, TARGET_FRAME_SIZE)
        out.append(cropped)
    return torch.stack(out, dim=0)


def load_video_tensors(path: Path, kind: VideoKind, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        visual: float tensor (1, 3, 45, 224, 224) — matches dataset eval (C,T,H,W).
        audio: float tensor (1, 1, 224, 224) log-mel resized — matches dataset.
    """
    v_clip, a_clip, info = _read_video_unified(path)
    if a_clip.shape[0] > 1:
        a_clip = torch.mean(a_clip, dim=0, keepdim=True)
    video_fps = float(info.get("video_fps", 30.0) or 30.0)
    audio_fps = float(info.get("audio_fps", AUDIO_SAMPLE_RATE) or AUDIO_SAMPLE_RATE)

    if kind == VideoKind.youtube:
        v_clip = _face_crop_clip_tchw(v_clip, device)
    else:
        v_clip = _spatial_resize_clip_tchw(v_clip)

    video = v_clip.float() / 255.0
    video = video.permute(1, 0, 2, 3).unsqueeze(0)

    a_clip = _to_device_float_audio(a_clip)
    if audio_fps != AUDIO_SAMPLE_RATE:
        a_clip = torchaudio.functional.resample(a_clip, int(audio_fps), AUDIO_SAMPLE_RATE)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=AUDIO_SAMPLE_RATE,
        n_mels=NUM_MEL_BINS,
        n_fft=2048,
        hop_length=512,
    )(a_clip)
    mel = torch.log(mel + 1e-9)
    mel = F.interpolate(mel.unsqueeze(0), size=TARGET_FRAME_SIZE, mode="bilinear", align_corners=False).squeeze(0)
    audio = mel.unsqueeze(0)

    return video, audio
