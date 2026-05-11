"""LangChain tool definitions for the audio-visual consistency analysis agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from backend.crud.inference_crud import run_av_prediction
from backend.crud.video_kind import VideoKind


@tool
async def validate_video_file(file_path: str) -> str:
    """
    Validate that the video file exists and is not empty.
    Returns a JSON string with validation details.
    """
    path = Path(file_path)
    result = {
        "exists": path.is_file(),
        "is_empty": path.stat().st_size == 0 if path.is_file() else True,
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "extension": path.suffix.lower(),
    }
    return json.dumps(result)


@tool
async def detect_audio_visual_consistency(
    file_path: str, video_kind: str
) -> str:
    """
    Run the audio-visual consistency detector on the uploaded video.
    Returns a JSON string with match_probability and interpretation.
    
    Args:
        file_path: Path to the video file
        video_kind: Either "youtube" or "dataset"
    """
    try:
        kind = VideoKind(video_kind)
        result = await run_av_prediction(Path(file_path), kind)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e), "match_probability": 0.0})


@tool
def analyze_consistency_score(score: float) -> str:
    """
    Interpret a match probability and return a risk assessment.
    
    Args:
        score: Match probability from 0.0 to 1.0
    """
    if score >= 0.75:
        assessment = "high_confidence_match"
        explanation = "The audio and visual streams are well aligned."
    elif score >= 0.5:
        assessment = "moderate_match"
        explanation = (
            "The alignment is borderline. This could indicate weak synchronization "
            "or domain shift (training vs. real-world data)."
        )
    else:
        assessment = "low_match"
        explanation = (
            "The audio and visual streams appear misaligned. This may indicate "
            "manipulation, severe domain shift, or technical issues."
        )

    return json.dumps(
        {
            "assessment": assessment,
            "explanation": explanation,
            "match_probability": score,
        }
    )


def get_tools():
    """Return list of all available tools for the agent."""
    return [
        validate_video_file,
        detect_audio_visual_consistency,
        analyze_consistency_score,
    ]
