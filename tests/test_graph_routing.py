"""Phase 3 — should_continue routing tests.

Pure unit tests against the state dict — no graph execution, no LLM.
should_continue is a regular function that accepts an AgentState dict and
returns a routing string, so it can be called directly.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import should_continue
from tests.conftest import ai_with_tool, make_state


# ---------------------------------------------------------------------------
# No tool calls → "done"
# ---------------------------------------------------------------------------

def test_no_tool_calls_routes_done():
    state = make_state(messages=[AIMessage(content="Here is your answer.")])
    assert should_continue(state) == "done"


def test_empty_content_and_no_tool_calls_routes_done():
    """Empty response (LLM returned nothing) still routes to done, not a crash."""
    state = make_state(messages=[AIMessage(content="")])
    assert should_continue(state) == "done"


def test_human_message_last_routes_done():
    """If the last message is not an AIMessage, no tool_calls attr → done."""
    state = make_state(messages=[HumanMessage(content="hello")])
    assert should_continue(state) == "done"


# ---------------------------------------------------------------------------
# Tool calls present, under cap, no loop → "continue"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iteration,max_iter", [
    (0, 10),
    (1, 10),
    (9, 10),
    (0, 1),
])
def test_tool_calls_under_cap_routes_continue(iteration, max_iter):
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "foo.py"})],
        iteration=iteration,
        max_iterations=max_iter,
        last_tool_call=None,
    )
    assert should_continue(state) == "continue"


def test_different_args_is_not_a_loop():
    """Same tool name but different args is NOT a loop."""
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "bar.py"})],
        iteration=2,
        max_iterations=10,
        last_tool_call={"name": "read_file", "args": {"path": "foo.py"}},
    )
    assert should_continue(state) == "continue"


def test_different_tool_is_not_a_loop():
    """Different tool name entirely is not a loop."""
    state = make_state(
        messages=[ai_with_tool("list_files", {"path": "."})],
        iteration=2,
        max_iterations=10,
        last_tool_call={"name": "read_file", "args": {"path": "."}},
    )
    assert should_continue(state) == "continue"


# ---------------------------------------------------------------------------
# Loop detection → "summarize"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,args", [
    ("read_file", {"path": "foo.py"}),
    ("search",    {"pattern": "def main", "path": "."}),
    ("list_files", {"path": "."}),
])
def test_repeated_tool_call_routes_summarize(tool_name, args):
    """Exact name+args repeat triggers loop detection → summarize."""
    state = make_state(
        messages=[ai_with_tool(tool_name, args)],
        iteration=1,
        max_iterations=10,
        last_tool_call={"name": tool_name, "args": args},
    )
    assert should_continue(state) == "summarize"


# ---------------------------------------------------------------------------
# Iteration cap → "summarize"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("iteration,max_iter", [
    (10, 10),   # exactly at cap
    (11, 10),   # over cap
    (100, 10),  # far over cap
    (1, 1),     # cap of 1
])
def test_iteration_cap_routes_summarize(iteration, max_iter):
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "x.py"})],
        iteration=iteration,
        max_iterations=max_iter,
        last_tool_call=None,  # not a loop — only cap triggers this
    )
    assert should_continue(state) == "summarize"


# ---------------------------------------------------------------------------
# Cap takes priority over loop (both conditions true simultaneously)
# ---------------------------------------------------------------------------

def test_cap_and_loop_both_true_routes_summarize():
    """When both loop and cap are detected, should_continue still returns 'summarize'."""
    same_call = {"name": "read_file", "args": {"path": "x.py"}}
    state = make_state(
        messages=[ai_with_tool("read_file", {"path": "x.py"})],
        iteration=10,
        max_iterations=10,
        last_tool_call=same_call,
    )
    assert should_continue(state) == "summarize"
