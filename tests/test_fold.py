"""Phase 4b — fold_branch integration tests.

Uses a real graph (with mock LLM) backed by a temp SQLite DB so we can
verify checkpoint state before and after folding.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.fold import fold_branch
from agent.branches import BranchManager
from tests.conftest import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_thread(graph, thread_id, messages):
    """Write messages into a thread's checkpoint via update_state."""
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"messages": messages})
    return config


def _get_messages(graph, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    return state.values.get("messages", []) if state else []


# ---------------------------------------------------------------------------
# Basic fold behaviour
# ---------------------------------------------------------------------------

def test_fold_appends_one_message_to_parent(mock_llm, branch_manager, test_graph):
    """Folding a child branch adds exactly one new message to the parent thread."""
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("exp", from_thread_id="default")
    branch_manager.set_active("exp")

    _seed_thread(graph, child_tid, [
        HumanMessage(content="exploring something"),
        AIMessage(content="I found it"),
    ])
    # Parent ("default") starts empty
    parent_before = _get_messages(graph, "default")

    mock_llm.invoke.return_value = AIMessage(content="summary of experiment")
    fold_branch(graph, branch_manager, mock_llm, "exp")

    parent_after = _get_messages(graph, "default")
    assert len(parent_after) == len(parent_before) + 1


def test_fold_child_checkpoint_unchanged(mock_llm, branch_manager, test_graph):
    """Folding must NOT delete or alter the child branch's checkpoint."""
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("side", from_thread_id="default")
    branch_manager.set_active("side")
    original = [HumanMessage(content="work done here")]
    _seed_thread(graph, child_tid, original)

    mock_llm.invoke.return_value = AIMessage(content="summary")
    fold_branch(graph, branch_manager, mock_llm, "side")

    child_after = _get_messages(graph, child_tid)
    assert len(child_after) == len(original)
    assert child_after[0].content == original[0].content


def test_fold_switches_active_to_parent(mock_llm, branch_manager, test_graph):
    """After fold, BranchManager.get_active() returns the parent branch name."""
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("work", from_thread_id="default")
    branch_manager.set_active("work")
    _seed_thread(graph, child_tid, [HumanMessage(content="done")])

    mock_llm.invoke.return_value = AIMessage(content="summary")
    fold_branch(graph, branch_manager, mock_llm, "work")

    assert branch_manager.get_active() == "main"


def test_fold_empty_branch_produces_message(mock_llm, branch_manager, test_graph):
    """Folding a branch with no messages still appends the placeholder message."""
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("empty", from_thread_id="default")
    branch_manager.set_active("empty")
    # Don't seed — leave child thread empty

    mock_llm.invoke.return_value = AIMessage(content="nothing happened")
    fold_branch(graph, branch_manager, mock_llm, "empty")

    parent_msgs = _get_messages(graph, "default")
    # Even for an empty branch, fold returns a "(no messages)" placeholder
    assert len(parent_msgs) == 1


def test_fold_no_parent_returns_error_string(mock_llm, branch_manager, test_graph):
    """fold_branch returns an error string when the branch has no parent."""
    graph, _ = test_graph
    # "main" has parent=None
    result = fold_branch(graph, branch_manager, mock_llm, "main")
    assert "cannot fold" in result.lower() or "no parent" in result.lower()


# ---------------------------------------------------------------------------
# Regression: fold must inject HumanMessage, NOT SystemMessage
# ---------------------------------------------------------------------------

def test_fold_injects_human_message_not_system_message(mock_llm, branch_manager, test_graph):
    """
    Regression guard for the SystemMessage injection bug.

    Before the fix, fold_branch injected SystemMessage into the parent
    checkpoint.  Claude's API then rejected the message history with
    'multiple non-consecutive system messages' on the next turn.

    After the fix, fold injects HumanMessage.  This test will FAIL if
    the injection type regresses back to SystemMessage.
    """
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("feature", from_thread_id="default")
    branch_manager.set_active("feature")
    _seed_thread(graph, child_tid, [
        HumanMessage(content="fix the bug"),
        AIMessage(content="fixed"),
    ])

    mock_llm.invoke.return_value = AIMessage(content="branch summary")
    fold_branch(graph, branch_manager, mock_llm, "feature")

    parent_msgs = _get_messages(graph, "default")
    assert parent_msgs, "expected at least one message in parent after fold"

    fold_msg = parent_msgs[-1]
    assert isinstance(fold_msg, HumanMessage), (
        f"fold injected {type(fold_msg).__name__} instead of HumanMessage — "
        "this will cause 'multiple non-consecutive system messages' errors from Claude."
    )


def test_fold_no_system_messages_in_parent_checkpoint(mock_llm, branch_manager, test_graph):
    """After fold, the parent checkpoint contains zero SystemMessages.

    The think node prepends its SystemMessage fresh each turn without
    persisting it, so a clean checkpoint should never have any SystemMessages.
    """
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("clean-test", from_thread_id="default")
    branch_manager.set_active("clean-test")
    _seed_thread(graph, child_tid, [HumanMessage(content="experimenting")])

    mock_llm.invoke.return_value = AIMessage(content="found something useful")
    fold_branch(graph, branch_manager, mock_llm, "clean-test")

    parent_msgs = _get_messages(graph, "default")
    system_msgs = [m for m in parent_msgs if isinstance(m, SystemMessage)]
    assert system_msgs == [], (
        f"Found {len(system_msgs)} SystemMessage(s) in parent checkpoint after fold; "
        "these will cause Claude API errors on the next turn."
    )


def test_fold_parent_usable_after_fold(mock_llm, branch_manager, test_graph):
    """After fold, streaming one turn on the parent does not raise.

    This is the full integration regression: if fold injected a SystemMessage,
    think() would build [SystemMessage(system_prompt), ..., SystemMessage(fold), ...]
    and Claude would reject the history.  With the mock LLM this checks the
    graph machinery path rather than the Claude API constraint, but it still
    verifies the graph doesn't raise during the turn.
    """
    graph, _ = test_graph

    child_tid = branch_manager.create_branch("temp", from_thread_id="default")
    branch_manager.set_active("temp")
    _seed_thread(graph, child_tid, [HumanMessage(content="some work")])

    mock_llm.invoke.return_value = AIMessage(content="summary text")
    fold_branch(graph, branch_manager, mock_llm, "temp")

    # Now stream one turn on the parent
    parent_tid = branch_manager.switch("main")
    parent_config = {"configurable": {"thread_id": parent_tid}}
    mock_llm.invoke.return_value = AIMessage(content="continuing from parent")

    events = list(graph.stream(
        {
            "messages": [HumanMessage(content="what did the branch find?")],
            "iteration": 0,
            "max_iterations": 5,
            "workspace_context": "",
            "last_tool_call": None,
            "memory": {"task": "", "files": [], "notes": []},
        },
        parent_config,
        stream_mode="updates",
    ))
    # No exception = pass
    assert events is not None
