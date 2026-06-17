"""Agent state — the single source of truth flowing through the graph."""

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


class AgentState(TypedDict):
    """
    State that flows between every node in the graph.

    - messages: The conversation history. add_messages is a LangGraph
      "reducer" — it appends new messages instead of replacing the list.
    - iteration: Current step count in the ReAct loop.
    - max_iterations: Safety limit .
    - workspace_context: Static string with git info, injected into system prompt.
    - memory: Distilled notes (like Raschka's memory dict).
    """
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    workspace_context: str
    memory: dict[str, Any]


def create_initial_state(
    user_message: str,
    workspace_context: str = "",
    max_iterations: int = 15,
) -> AgentState:
    """
    Create a fresh AgentState for a new user turn.
    
    """
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
    }