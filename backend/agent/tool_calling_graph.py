"""Tool-calling agent graph for audio-visual consistency analysis."""
from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from backend.agent.tools import get_tools


class ToolCallingState(TypedDict, total=False):
    saved_path: str
    video_kind: str
    file_name: str
    file_size_bytes: int
    messages: list[dict[str, str]]
    final_response: str


def _load_llm():
    """Load ChatOllama when the local Ollama server is available."""
    if os.environ.get("OLLAMA_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        return None

    model_name = os.environ.get("OLLAMA_MODEL", "llama3.1")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
    return llm.bind_tools(get_tools())


def input_node(state: ToolCallingState) -> ToolCallingState:
    """Start the conversation with context about the uploaded video."""
    path = Path(state["saved_path"])
    size_mb = state["file_size_bytes"] / (1024 * 1024)
    
    initial_prompt = (
        f"I've uploaded a video file '{state['file_name']}' "
        f"(size: {size_mb:.1f} MB, kind: {state['video_kind']}). "
        f"Please analyze it for audio-visual consistency. "
        f"Start by validating the file, then run the detector, "
        f"then analyze the results and provide a clear recommendation."
    )
    
    state["messages"] = [{"role": "user", "content": initial_prompt}]
    return state


async def llm_node(state: ToolCallingState) -> ToolCallingState:
    """Call the LLM with tool binding."""
    llm = _load_llm()

    if llm is None:
        state["final_response"] = (
            "Ollama is not configured. Install/start Ollama, run `ollama pull "
            "llama3.1`, ensure the Ollama server is reachable at "
            "OLLAMA_BASE_URL (default: http://localhost:11434), then retry."
        )
        return state

    messages = []
    for msg in state.get("messages", []):
        role = msg.get("role")
        if role == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif role == "assistant":
            messages.append(
                AIMessage(
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls", []),
                )
            )
        elif role == "tool":
            messages.append(
                ToolMessage(
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )

    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        # Record the error in the conversation instead of raising
        state["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "Ollama call failed: "
                    f"{e}. Confirm Ollama is running (`ollama serve`) and that "
                    "the model is available (`ollama pull llama3.1`)."
                ),
                "tool_calls": [],
            }
        )
        # Also set final_response so the API can return a graceful message
        state["final_response"] = (
            "Ollama call failed: "
            f"{e}. Confirm Ollama is running (`ollama serve`) and that "
            "the model is available (`ollama pull llama3.1`)."
        )
        return state

    state["messages"].append(
        {
            "role": "assistant",
            "content": response.content,
            "tool_calls": getattr(response, "tool_calls", []),
        }
    )

    return state


def should_continue(state: ToolCallingState) -> Literal["tools", "end"]:
    """Check if the last message has tool calls."""
    if not state.get("messages"):
        return "end"
    
    last_msg = state["messages"][-1]
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        return "tools"
    
    return "end"


async def tools_node(state: ToolCallingState) -> ToolCallingState:
    """Execute tool calls from the LLM."""
    if not state.get("messages"):
        return state

    last_msg = state["messages"][-1]
    tool_calls = last_msg.get("tool_calls", [])

    if not tool_calls:
        return state

    available_tools = {tool.name: tool for tool in get_tools()}

    # Execute each tool call
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args", {})

        if tool_name not in available_tools:
            result = json.dumps({"error": f"Unknown tool: {tool_name}"})
        else:
            try:
                tool = available_tools[tool_name]
                if inspect.iscoroutinefunction(tool.func):
                    result = await tool.func(**tool_input)
                else:
                    result = tool.func(**tool_input)
            except Exception as e:
                result = json.dumps({"error": str(e)})

        state["messages"].append(
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.get("id", ""),
                "name": tool_name,
            }
        )

    return state


def final_node(state: ToolCallingState) -> ToolCallingState:
    """Extract the final response from the agent."""
    if not state.get("messages"):
        state["final_response"] = "No response generated."
        return state
    
    # Find the last assistant message with content (not tool calls)
    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant" and msg.get("content"):
            state["final_response"] = msg["content"]
            break
    
    if "final_response" not in state:
        state["final_response"] = "Analysis complete."
    
    return state


def build_tool_calling_graph():
    """Build the tool-calling agent graph."""
    workflow = StateGraph(ToolCallingState)

    workflow.add_node("input", input_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("final", final_node)

    workflow.add_edge(START, "input")
    workflow.add_edge("input", "llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": "final",
        },
    )
    workflow.add_edge("tools", "llm")
    workflow.add_edge("final", END)

    return workflow.compile()
