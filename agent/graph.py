"""Graph assembly — wires nodes into a LangGraph StateGraph."""

from __future__ import annotations
from functools import partial
import yaml


from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import think, act, should_continue
from tools.registry import  get_builtin_tools
from providers.factory import create_llm
from mcp.client import MCPManager
from mcp.tool_bridge import create_mcp_langchain_tools
def build_graph(
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    max_iterations: int = 15,
    extra_tools=None,     
    **llm_kwargs,
):
    """
    Build the agent graph.

    Returns:
        (compiled_graph, config_dict)
    """
    # 1. Create the LLM
    llm = create_llm(provider, model, **llm_kwargs)


   

    # 2. Get tools and bind them to the LLM
    builtin_tools = get_builtin_tools()
    
    builtin_tools = builtin_tools + (extra_tools or [])
    llm_with_tools = llm.bind_tools(builtin_tools)

    # 3. Build a tool lookup map: name -> tool function
    tool_map = {tool.name: tool for tool in builtin_tools}

    # 4. Create partial node functions (bake in dependencies)
    think_node = partial(think, llm_with_tools=llm_with_tools)
    act_node = partial(act, tool_map=tool_map)

    # 5. Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)

    # Set entry point
    graph.set_entry_point("think")

    # Add edges
    # thinchk always goes to should_continue eck
    graph.add_conditional_edges(
        "think", #source node 
        should_continue,
        {
            "continue": "act",     # if should_continue, use a tool → execute it
            "done": END,           # LLM gave final answer → stop
        }
    )

    # After acting, always go back to think (the ReAct loop)
    graph.add_edge("act", "think") 

    # 6. Compile with checkpointer for session persistence
    checkpointer = MemorySaver()  # In-memory; swap to SqliteSaver for disk persistence
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled