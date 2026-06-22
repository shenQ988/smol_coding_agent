"""Graph assembly — wires nodes into a LangGraph StateGraph."""

from __future__ import annotations
from functools import partial
from pathlib import Path
import sqlite3

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import AgentState
from agent.nodes import think, act, should_continue, summarize
from tools.registry import  get_builtin_tools
from providers.factory import create_llm
from mcp_integration.client import MCPManager
from mcp_integration.tool_bridge import create_mcp_langchain_tools
from agent.cost_tracker import CostTracker
from context.skills import SkillStore

def build_graph(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    max_iterations: int = 15,
    extra_tools=None,
    llm=None,
    cost_tracker=None,
    db_path: str = ".checkpoints.db",
    **llm_kwargs
):
    """
    Build the agent graph.

    Returns:
        (compiled_graph, sqlite3.Connection) — caller must close the connection on exit.
    """
    # 1. Create the LLM
    llm = llm or create_llm(provider, model, **llm_kwargs)


   

    # 2. Get tools and bind them to the LLM
    builtin_tools = get_builtin_tools()
    
    builtin_tools = builtin_tools + (extra_tools or [])
    llm_with_tools = llm.bind_tools(builtin_tools)

    # 3. Build a tool lookup map: name -> tool function
    tool_map = {tool.name: tool for tool in builtin_tools}

    # 4. Create partial node functions (bake in dependencies)
    skill_store = SkillStore(Path(__file__).parent.parent / "skills")
    think_node    = partial(think,     llm_with_tools=llm_with_tools, skill_store=skill_store, cost_tracker=cost_tracker)
    act_node      = partial(act,       tool_map=tool_map)
    summarize_node = partial(summarize, llm=llm)

    # 5. Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("think",     think_node)
    graph.add_node("act",       act_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("think")

    #
    # Edges:
    #
    #   think ──► should_continue ──► "continue"  → act → think   (ReAct loop)
    #                               ──► "done"      → END          (normal finish)
    #                               ──► "summarize" → summarize    (hit cap / loop)
    #
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "continue":  "act",
            "done":      END,
            "summarize": "summarize",
        }
    )
    graph.add_edge("act",       "think")
    graph.add_edge("summarize", END)

    # 6. Compile with checkpointer for session persistence
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled, conn