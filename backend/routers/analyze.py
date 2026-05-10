from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.agent.analysis_graph import build_analysis_graph
from backend.agent.tool_calling_graph import build_tool_calling_graph
from backend.config import MAX_UPLOAD_MB
from backend.crud.inference_crud import save_upload
from backend.crud.video_kind import VideoKind

router = APIRouter(prefix="/api", tags=["analysis"])

_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_GRAPH = build_analysis_graph()
_TOOL_CALLING_GRAPH = build_tool_calling_graph()


def _suffix(filename: str | None) -> str:
    if not filename:
        return ".mp4"
    p = Path(filename).suffix.lower()
    return p if p in {".mp4", ".webm", ".mov", ".mkv", ".avi"} else ".mp4"


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    video_kind: str = Form(..., description='Either "youtube" or "dataset".'),
    agent_mode: str = Query(
        "simple",
        description='Either "simple" (rule-based) or "tool_calling" (LLM-based)',
    ),
):
    raw = await file.read()
    if len(raw) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit.")

    try:
        kind = VideoKind(video_kind)
    except ValueError:
        raise HTTPException(400, 'video_kind must be "youtube" or "dataset".')

    path = await save_upload(raw, _suffix(file.filename))
    try:
        if agent_mode == "tool_calling":
            result = await _TOOL_CALLING_GRAPH.ainvoke(
                {
                    "saved_path": str(path),
                    "video_kind": kind.value,
                    "file_name": file.filename or path.name,
                    "file_size_bytes": len(raw),
                }
            )
            # Extract the final response and earlier messages
            messages = result.get("messages", [])
            final_response = result.get("final_response", "")
            
            return {
                "agent_mode": "tool_calling",
                "final_response": final_response,
                "messages": messages,
                "video_kind": kind.value,
            }
        else:
            result = await _GRAPH.ainvoke(
                {
                    "saved_path": str(path),
                    "video_kind": kind.value,
                    "file_name": file.filename or path.name,
                    "file_size_bytes": len(raw),
                }
            )
            prediction = result.get("prediction", {})
            validation = result.get("validation", {})

            return {
                "agent_mode": "simple",
                "match_probability": prediction.get("match_probability", 0.0),
                "interpretation": prediction.get("interpretation", ""),
                "video_kind": prediction.get("video_kind", kind.value),
                "risk_level": result.get("risk_level", "unknown"),
                "recommendation": result.get("recommendation", ""),
                "agent_summary": result.get("agent_summary", ""),
                "validation": validation,
            }
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Agent analysis failed: {e}") from e
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass