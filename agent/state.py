"""Agent state — the single source of truth flowing through the graph."""

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict):
    """
    State that flows between every node in the graph.

    - messages: The conversation history. add_messages appends instead of replacing.
    - iteration: Per-turn ReAct step count, reset each user message.
    - max_iterations: Per-turn safety cap.
    - workspace_context: Static string with git info, injected into system prompt.
    - memory: Distilled notes persisted across turns.
    - last_tool_call: The last tool call executed, used for loop detection.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    workspace_context: str
    memory: dict[str, Any]
    last_tool_call: dict[str, Any] | None


def build_turn_input(
    graph,
    config: dict,
    user_message: str,
    workspace_context: str = "",
    max_iterations: int = 15,
) -> tuple[dict, bool]:
    """Build per-turn stream_input for graph.stream().

    Checks whether the thread's checkpoint already contains max_iterations.
    If not (fresh thread or branch seeded without full init), includes
    max_iterations and a blank memory in the returned dict so the graph
    initialises them.  Callers should pass the returned dict directly to
    graph.stream() — never hard-code this logic elsewhere.

    Returns (stream_input, needs_init).
    """
    existing = graph.get_state(config)
    needs_init = not existing or "max_iterations" not in (existing.values or {})
    stream_input: dict = {
        "messages": [HumanMessage(content=user_message)],
        "iteration": 0,
        "workspace_context": workspace_context,
        "last_tool_call": None,
    }
    if needs_init:
        stream_input["max_iterations"] = max_iterations
        stream_input["memory"] = {"task": "", "files": [], "notes": []}
    return stream_input, needs_init


def create_initial_state(
    user_message: str,
    workspace_context: str = "",
    max_iterations: int = 15,
) -> AgentState:
    return {
        "messages": [HumanMessage(content=user_message)],
        "iteration": 0,
        "max_iterations": max_iterations,
        "workspace_context": workspace_context,
        "memory": {
            "task": "",
            "files": [],
            "notes": [],
        },
        "last_tool_call": None,
    }