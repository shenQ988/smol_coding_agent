"""Phase 5 — Memory persistence regression tests.

The key invariant: per-turn fields (iteration, workspace_context, last_tool_call)
are reset each turn via stream_input, but cross-turn fields (memory, max_iterations)
must survive from the checkpoint and must NOT be included in stream_input for
non-new threads.

build_turn_input (agent/state.py) is the single source of truth for this logic —
both main.py and these tests import it from there, so a regression in either
place is caught by the same test.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes import act
from agent.state import build_turn_input
from tests.conftest import ai_with_tool, make_state


# ---------------------------------------------------------------------------
# Core regression: memory survives turn 2
# ---------------------------------------------------------------------------

def test_memory_not_reset_on_second_turn(mock_llm, test_graph):
    """
    Regression guard for the memory-reset bug.

    Bug: main.py included 'memory' in every stream_input, wiping the
    agent's accumulated memory on each turn.
    Fix: only include 'memory' when is_new_thread is True.

    This test seeds memory after turn 1 and verifies it survives turn 2.
    If main.py is changed to always pass 'memory' in stream_input,
    this test will catch the regression.
    """
    graph, _ = test_graph
    config = {"configurable": {"thread_id": "memory-persist-test"}}

    # --- Turn 1 ---
    stream_input_1, is_new = build_turn_input(graph, config, "hello")
    assert is_new, "first turn must be recognised as a new thread"
    for _ in graph.stream(stream_input_1, config, stream_mode="updates"):
        pass

    # Simulate the agent having written a file (act node updates memory).
    # We inject this directly to avoid needing a risky tool call with interrupt.
    graph.update_state(config, {
        "memory": {"task": "refactor auth", "files": ["auth.py"], "notes": ["key note"]}
    })

    # --- Turn 2 stream_input construction ---
    stream_input_2, is_new_2 = build_turn_input(graph, config, "continue")
    assert not is_new_2, "second turn must NOT be recognised as a new thread"
    assert "memory" not in stream_input_2, (
        "memory must NOT appear in turn-2 stream_input — "
        "if it does, it will overwrite the checkpoint's memory"
    )

    # --- Turn 2 execution ---
    for _ in graph.stream(stream_input_2, config, stream_mode="updates"):
        pass

    # --- Verify memory survived ---
    final_state = graph.get_state(config)
    mem = final_state.values.get("memory", {})
    assert mem.get("files") == ["auth.py"], (
        f"memory['files'] was reset to {mem.get('files')!r} — "
        "the memory-reset bug has reappeared"
    )
    assert mem.get("task") == "refactor auth"


def test_max_iterations_survives_second_turn(mock_llm, test_graph):
    """max_iterations set in turn 1 must be preserved into turn 2."""
    graph, _ = test_graph
    config = {"configurable": {"thread_id": "maxiter-persist-test"}}
    custom_max = 7

    stream_input_1, _ = build_turn_input(graph, config, "task", max_iterations=custom_max)
    for _ in graph.stream(stream_input_1, config, stream_mode="updates"):
        pass

    stream_input_2, is_new = build_turn_input(graph, config, "next", max_iterations=custom_max)
    assert not is_new
    assert "max_iterations" not in stream_input_2

    for _ in graph.stream(stream_input_2, config, stream_mode="updates"):
        pass

    state = graph.get_state(config)
    assert state.values["max_iterations"] == custom_max


# ---------------------------------------------------------------------------
# Notes cap regression: act node caps memory['notes'] to 20
# ---------------------------------------------------------------------------

def test_notes_cap_in_act_node():
    """act() caps memory['notes'] at 20 entries after tool execution."""
    from unittest.mock import MagicMock

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "tool result"

    state = make_state(
        messages=[ai_with_tool("mock_tool", {})],
        memory={
            "task": "",
            "files": [],
            "notes": [f"note {i}" for i in range(25)],
        },
    )
    result = act(state, tool_map={"mock_tool": mock_tool})
    assert "memory" in result
    assert len(result["memory"]["notes"]) <= 20


def test_notes_cap_keeps_most_recent():
    """act() keeps the LAST 20 notes (not the first 20)."""
    from unittest.mock import MagicMock

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "ok"

    notes = [f"note {i}" for i in range(25)]  # 0..24
    state = make_state(
        messages=[ai_with_tool("mock_tool", {})],
        memory={"task": "", "files": [], "notes": notes},
    )
    result = act(state, tool_map={"mock_tool": mock_tool})
    kept = result["memory"]["notes"]
    assert kept == notes[-20:]  # notes 5..24


def test_notes_cap_under_limit_unchanged():
    """act() leaves notes alone when there are ≤ 20."""
    from unittest.mock import MagicMock

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "ok"

    notes = ["a", "b", "c"]
    state = make_state(
        messages=[ai_with_tool("mock_tool", {})],
        memory={"task": "", "files": [], "notes": notes},
    )
    result = act(state, tool_map={"mock_tool": mock_tool})
    assert result["memory"]["notes"] == notes


# ---------------------------------------------------------------------------
# is_new_thread logic — edge cases
# ---------------------------------------------------------------------------

def test_is_new_thread_true_for_fresh_thread(test_graph):
    """A thread that has never been used is recognised as new."""
    graph, _ = test_graph
    config = {"configurable": {"thread_id": "brand-new-xyz"}}
    _, is_new = build_turn_input(graph, config, "hi")
    assert is_new


def test_is_new_thread_false_after_first_turn(mock_llm, test_graph):
    """After one turn, the thread is no longer recognised as new."""
    graph, _ = test_graph
    config = {"configurable": {"thread_id": "second-turn-test"}}
    si, _ = build_turn_input(graph, config, "first")
    for _ in graph.stream(si, config, stream_mode="updates"):
        pass
    _, is_new = build_turn_input(graph, config, "second")
    assert not is_new


def test_is_new_thread_true_after_partial_seed(branch_manager, test_graph):
    """
    Belt test: if a checkpoint somehow has messages but no max_iterations
    (e.g. manual update_state, future code path), the is_new_thread check
    must still detect the gap and include max_iterations in stream_input.

    This scenario should no longer arise from /branch (source fix seeds the
    full state), but the check in main.py and the defensive .get() in
    should_continue both guard against it anyway.
    """
    graph, _ = test_graph

    new_tid = branch_manager.create_branch("partial-seed", from_thread_id="default")
    seed_config = {"configurable": {"thread_id": new_tid}}
    # Deliberately omit max_iterations — simulates old broken path or manual misuse.
    graph.update_state(seed_config, {
        "messages": [HumanMessage(content="[Branched from 'main']\nsome summary")]
    })

    _, is_new = build_turn_input(graph, seed_config, "first task in branch")
    assert is_new, (
        "A checkpoint with messages but no max_iterations must be treated as new — "
        "otherwise should_continue will crash with KeyError: 'max_iterations'"
    )


def test_branch_command_seeds_max_iterations(mock_llm, branch_manager, test_graph):
    """
    Source fix: commands/branch.run() must write max_iterations into the new
    thread's checkpoint so should_continue never sees it absent.
    """
    import commands.branch as branch_cmd

    graph, _ = test_graph
    parent_config = {"configurable": {"thread_id": "default"}}

    # Give the parent a fully initialised checkpoint.
    for _ in graph.stream(
        {
            "messages": [HumanMessage(content="parent task")],
            "iteration": 0,
            "max_iterations": 7,
            "workspace_context": "",
            "last_tool_call": None,
            "memory": {"task": "", "files": [], "notes": []},
        },
        parent_config,
        stream_mode="updates",
    ):
        pass

    result = branch_cmd.run(
        args="child",
        branch_manager=branch_manager,
        graph_config=parent_config,
        graph=graph,
        llm=mock_llm,
    )
    assert "__SWITCH_BRANCH__" in result

    new_tid = result.removeprefix("__SWITCH_BRANCH__").partition(":")[0]
    new_config = {"configurable": {"thread_id": new_tid}}
    state = graph.get_state(new_config)

    assert "max_iterations" in state.values, (
        "branch.run() must seed max_iterations into the new checkpoint"
    )
    assert state.values["max_iterations"] == 7, (
        "branch should inherit max_iterations from parent"
    )


def test_first_turn_after_branch_no_key_error(mock_llm, branch_manager, test_graph):
    """
    End-to-end regression: create a branch via branch.run(), then stream
    one turn on it.  Must not raise KeyError: 'max_iterations'.
    """
    import commands.branch as branch_cmd

    graph, _ = test_graph
    parent_config = {"configurable": {"thread_id": "default"}}

    for _ in graph.stream(
        {
            "messages": [HumanMessage(content="initial work")],
            "iteration": 0,
            "max_iterations": 10,
            "workspace_context": "",
            "last_tool_call": None,
            "memory": {"task": "", "files": [], "notes": []},
        },
        parent_config,
        stream_mode="updates",
    ):
        pass

    branch_cmd.run(
        args="e2e-branch",
        branch_manager=branch_manager,
        graph_config=parent_config,
        graph=graph,
        llm=mock_llm,
    )

    new_tid = branch_manager.switch("e2e-branch")
    new_config = {"configurable": {"thread_id": new_tid}}

    # build_stream_input must treat this as NOT new (source fix seeded everything).
    stream_input, is_new = build_turn_input(graph, new_config, "first branch task")
    assert not is_new, "branch checkpoint is fully initialised — should not be new"

    # Actually stream one turn — no KeyError allowed.
    mock_llm.invoke.return_value = AIMessage(content="on it")
    events = list(graph.stream(stream_input, new_config, stream_mode="updates"))
    assert events  # at least one event emitted, no exception
