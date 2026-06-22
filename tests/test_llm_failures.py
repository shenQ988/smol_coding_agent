"""Phase 2 — LLM failure mode tests.

We characterise current behaviour without modifying source files.
xfail tests document gaps where exceptions should be caught but aren't.
"""

import time
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import act, should_continue, summarize, think
from tests.conftest import ai_with_tool, make_state


# ---------------------------------------------------------------------------
# 1. LLM raises in think node
# ---------------------------------------------------------------------------

def test_think_llm_exception_returns_error_message(mock_llm):
    """think() catches LLM exceptions and returns an error AIMessage instead of raising."""
    mock_llm.invoke.side_effect = RuntimeError("rate limit exceeded")
    state = make_state(messages=[HumanMessage(content="hello")])
    result = think(state, llm_with_tools=mock_llm)
    msgs = result.get("messages", [])
    assert msgs, "expected at least one message in result"
    assert any("error" in str(m.content).lower() for m in msgs)


def test_think_llm_exception_has_no_tool_calls(mock_llm):
    """Error AIMessage from think has no tool_calls → should_continue routes to done."""
    mock_llm.invoke.side_effect = RuntimeError("network error")
    state = make_state(messages=[HumanMessage(content="hello")])
    result = think(state, llm_with_tools=mock_llm)
    msgs = result.get("messages", [])
    assert not getattr(msgs[0], "tool_calls", None)
    assert should_continue({**state, "messages": msgs}) == "done"


def test_think_llm_exception_handled_by_graph(mock_llm, test_graph):
    """graph.stream() completes normally when think's LLM call fails (no raise)."""
    graph, _ = test_graph
    mock_llm.invoke.side_effect = RuntimeError("auth error")
    config = {"configurable": {"thread_id": "fail-think-graph"}}
    stream_input = {
        "messages": [HumanMessage(content="hello")],
        "iteration": 0,
        "max_iterations": 5,
        "workspace_context": "",
        "last_tool_call": None,
        "memory": {"task": "", "files": [], "notes": []},
    }
    events = list(graph.stream(stream_input, config, stream_mode="updates"))
    all_msgs = [
        m
        for event in events
        for node_output in event.values()
        if isinstance(node_output, dict)
        for m in node_output.get("messages", [])
    ]
    assert any("error" in str(m.content).lower() for m in all_msgs)


# ---------------------------------------------------------------------------
# 2. act node — unknown tool name → error ToolMessage (not a raise)
# ---------------------------------------------------------------------------

def test_act_unknown_tool_returns_error_tool_message():
    """act() returns an error ToolMessage for tool names not in tool_map."""
    state = make_state(messages=[ai_with_tool("nonexistent_tool", {})])
    result = act(state, tool_map={})
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert "unknown tool" in msgs[0].content.lower()


def test_act_missing_required_args_returns_error_tool_message():
    """act() wraps validation errors (missing required args) in ToolMessages."""
    from tools.filesystem import read_file
    # read_file requires 'path'; passing {} triggers a pydantic ValidationError
    # inside the ThreadPoolExecutor, which the outer except catches.
    state = make_state(messages=[ai_with_tool("read_file", {})])
    result = act(state, tool_map={"read_file": read_file})
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert "error" in msgs[0].content.lower()


# ---------------------------------------------------------------------------
# 3. Loop detection — same tool+args twice → should_continue returns "summarize"
# ---------------------------------------------------------------------------

def test_loop_detection_routes_to_summarize():
    """should_continue returns 'summarize' when the same tool call is repeated."""
    repeated_call = {"name": "read_file", "args": {"path": "foo.py"}}
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "foo.py"})],
        iteration=1,
        last_tool_call=repeated_call,  # same as what's about to run
    )
    assert should_continue(state) == "summarize"


def test_loop_detection_ignores_different_args():
    """should_continue does NOT flag as loop when args differ."""
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "bar.py"})],
        iteration=1,
        last_tool_call={"name": "read_file", "args": {"path": "foo.py"}},
    )
    assert should_continue(state) == "continue"


# ---------------------------------------------------------------------------
# 4. Iteration cap → should_continue returns "summarize"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iteration,max_iter,expected", [
    (10, 10, "summarize"),   # at cap
    (11, 10, "summarize"),   # over cap
    (9,  10, "continue"),    # under cap
    (0,  10, "continue"),    # fresh
])
def test_iteration_cap(iteration, max_iter, expected):
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "x.py"})],
        iteration=iteration,
        max_iterations=max_iter,
    )
    assert should_continue(state) == expected


# ---------------------------------------------------------------------------
# 5. Tool timeout in act node (ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def test_act_tool_timeout_returns_error_message(monkeypatch):
    """act() returns a timeout error ToolMessage when a tool exceeds TOOL_TIMEOUT."""
    import agent.nodes as nodes_module
    monkeypatch.setattr(nodes_module, "TOOL_TIMEOUT", 0.05)

    slow_tool = MagicMock()
    slow_tool.invoke.side_effect = lambda _args: (time.sleep(2), "done")[1]

    state = make_state(messages=[ai_with_tool("slow_tool", {})])
    result = act(state, tool_map={"slow_tool": slow_tool})
    msgs = result["messages"]
    assert len(msgs) == 1
    assert "timed out" in msgs[0].content.lower()


def test_think_has_no_llm_timeout(mock_llm):
    """Documentation: think() has no timeout on the LLM call itself.

    Verified by inspection: agent/nodes.py:think() calls llm_with_tools.invoke()
    with no ThreadPoolExecutor or asyncio.wait_for wrapper.  A hung LLM call
    blocks the REPL indefinitely.  This test just makes the gap explicit.
    """
    import inspect
    import agent.nodes as nodes_module
    source = inspect.getsource(nodes_module.think)
    assert "ThreadPoolExecutor" not in source, (
        "think() now has a timeout — remove this documentation test"
    )
    assert "wait_for" not in source, (
        "think() now has an async timeout — remove this documentation test"
    )


# ---------------------------------------------------------------------------
# 6. Summarization LLM failure — compact and fold don't catch it
# ---------------------------------------------------------------------------

def test_compact_llm_failure_returns_error_string(mock_llm, test_graph):
    """compact surfaces a clean error string instead of crashing the REPL."""
    import commands.compact as compact_cmd
    graph, _ = test_graph
    config = {"configurable": {"thread_id": "compact-fail"}}
    # Seed >4 messages so compact doesn't short-circuit
    graph.update_state(config, {
        "messages": [HumanMessage(content=f"msg {i}") for i in range(5)],
    })
    mock_llm.invoke.side_effect = Exception("API down")

    result = compact_cmd.run(graph=graph, graph_config=config, llm=mock_llm)
    # Should reach here with a graceful error; currently raises instead
    assert isinstance(result, str)
    assert "error" in result.lower()


def test_fold_llm_failure_returns_error_string(mock_llm, branch_manager, test_graph):
    """fold_branch surfaces a clean error string instead of crashing the REPL."""
    from agent.fold import fold_branch
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("exp", from_thread_id="default")
    branch_manager.set_active("exp")
    child_config = {"configurable": {"thread_id": child_tid}}
    graph.update_state(child_config, {"messages": [HumanMessage(content="experiment")]})

    mock_llm.invoke.side_effect = Exception("summarise failed")

    result = fold_branch(graph, branch_manager, mock_llm, "exp")
    # Should reach here with graceful error; currently raises instead
    assert isinstance(result, str)
    assert "error" in result.lower()
