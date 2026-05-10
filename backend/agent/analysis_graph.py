from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.crud.inference_crud import run_av_prediction
from backend.crud.video_kind import VideoKind


class AnalysisState(TypedDict, total=False):
    saved_path: str
    video_kind: str
    file_name: str
    file_size_bytes: int
    validation: dict[str, Any]
    prediction: dict[str, Any]
    risk_level: str
    recommendation: str
    agent_summary: str


def _load_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0)


def preflight_node(state: AnalysisState) -> AnalysisState:
    path = Path(state["saved_path"])
    kind = VideoKind(state["video_kind"])
    size_bytes = state.get("file_size_bytes", 0)

    validation: dict[str, Any] = {
        "exists": path.is_file(),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "kind": kind.value,
        "is_valid": path.is_file() and size_bytes > 0,
        "issues": [],
    }

    if not validation["exists"]:
        validation["issues"].append("uploaded file was not found on disk")
    if size_bytes <= 0:
        validation["issues"].append("uploaded file is empty")
        validation["is_valid"] = False

    if path.suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
        validation["issues"].append("file extension was normalized to a supported video type")

    state["validation"] = validation
    return state


def route_after_preflight(state: AnalysisState) -> Literal["predict", "fail"]:
    if state["validation"]["is_valid"]:
        return "predict"
    return "fail"


async def predict_node(state: AnalysisState) -> AnalysisState:
    kind = VideoKind(state["video_kind"])
    prediction = await run_av_prediction(Path(state["saved_path"]), kind)
    state["prediction"] = prediction
    return state


def fail_node(state: AnalysisState) -> AnalysisState:
    state["prediction"] = {
        "match_probability": 0.0,
        "interpretation": "The uploaded file could not be analyzed.",
        "video_kind": state["video_kind"],
    }
    return state


def assess_risk_node(state: AnalysisState) -> AnalysisState:
    probability = float(state["prediction"].get("match_probability", 0.0))
    issues = state["validation"].get("issues", [])

    if issues:
        risk_level = "needs_review"
        recommendation = "Fix the input issues before trusting the score."
    elif probability >= 0.75:
        risk_level = "high_consistency"
        recommendation = "The clip looks well aligned."
    elif probability >= 0.45:
        risk_level = "uncertain"
        recommendation = "The result is borderline and could reflect domain shift or weak sync."
    else:
        risk_level = "low_consistency"
        recommendation = "The clip may contain a mismatch or manipulation."

    state["risk_level"] = risk_level
    state["recommendation"] = recommendation
    return state


async def summarize_node(state: AnalysisState) -> AnalysisState:
    probability = float(state["prediction"].get("match_probability", 0.0))
    issues = state["validation"].get("issues", [])
    llm = _load_llm()

    if llm is None:
        if issues:
            summary = (
                f"The agent flagged {len(issues)} input issue(s) before trusting the detector. "
                f"The current score is {probability:.3f}. {state['recommendation']}"
            )
        else:
            summary = (
                f"The detector returned a match probability of {probability:.3f}. "
                f"Risk level: {state['risk_level']}. {state['recommendation']}"
            )
        state["agent_summary"] = summary
        return state

    prompt = (
        "You are explaining an audio-visual consistency analysis to a user. "
        "Write 2-3 concise sentences. Mention the score, the risk level, "
        "any validation issues, and one practical next step.\n\n"
        f"Validation: {state['validation']}\n"
        f"Prediction: {state['prediction']}\n"
        f"Risk level: {state['risk_level']}\n"
        f"Recommendation: {state['recommendation']}"
    )

    try:
        response = await llm.ainvoke(prompt)
        summary = getattr(response, "content", None) or str(response)
    except Exception:
        summary = (
            f"The detector returned a match probability of {probability:.3f}. "
            f"Risk level: {state['risk_level']}. {state['recommendation']}"
        )

    state["agent_summary"] = summary
    return state


def build_analysis_graph():
    workflow = StateGraph(AnalysisState)

    workflow.add_node("preflight", preflight_node)
    workflow.add_node("predict", predict_node)
    workflow.add_node("fail", fail_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "preflight")
    workflow.add_conditional_edges(
        "preflight",
        route_after_preflight,
        {
            "predict": "predict",
            "fail": "fail",
        },
    )
    workflow.add_edge("predict", "assess_risk")
    workflow.add_edge("fail", "assess_risk")
    workflow.add_edge("assess_risk", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()
