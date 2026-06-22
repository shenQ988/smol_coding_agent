"""Shared fixtures for the smol agent test suite."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.branches import BranchManager


@pytest.fixture
def mock_llm():
    """Fake LLM whose .invoke() returns a canned AIMessage (no tool calls).

    Tests can override behaviour per-test:
        mock_llm.invoke.return_value = AIMessage(content="...", tool_calls=[...])
        mock_llm.invoke.side_effect = Exception("rate limit")
        mock_llm.invoke.side_effect = [AIMessage(...), AIMessage(...)]   # sequence
    """
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="done")
    # bind_tools() is called once in build_graph; return the same mock so that
    # llm_with_tools.invoke == llm.invoke in tests.
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def tmp_checkpointer(tmp_path):
    """Real SqliteSaver backed by a temp file — same setup as production."""
    conn = sqlite3.connect(str(tmp_path / "test.db"), check_same_thread=False)
    cs = SqliteSaver(conn)
    yield cs
    conn.close()


@pytest.fixture
def tmp_workspace(tmp_path):
    """Empty temp directory to stand in for WORKSPACE_ROOT in tool tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def branch_manager(tmp_path):
    """BranchManager pointed at a temp .branches.json (not the real one)."""
    return BranchManager(index_path=tmp_path / ".branches.json")


@pytest.fixture
def test_graph(mock_llm, tmp_path):
    """Compiled LangGraph graph backed by a temp SQLite DB, using mock_llm.

    Yields (graph, conn).  The connection is closed in teardown.
    """
    from agent.graph import build_graph
    graph, conn = build_graph(
        provider="anthropic",
        model="claude-sonnet-4-6",
        llm=mock_llm,
        db_path=str(tmp_path / "test.db"),
    )
    yield graph, conn
    conn.close()


# ---------------------------------------------------------------------------
# Helpers used across multiple test modules
# ---------------------------------------------------------------------------

def make_state(**overrides):
    """Return a minimal AgentState dict, with optional field overrides."""
    base = {
        "messages": [],
        "iteration": 0,
        "max_iterations": 10,
        "workspace_context": "",
        "memory": {"task": "", "files": [], "notes": []},
        "last_tool_call": None,
    }
    base.update(overrides)
    return base


def ai_with_tool(name: str, args: dict, tc_id: str = "tc1") -> AIMessage:
    """AIMessage carrying a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": tc_id, "type": "tool_call"}],
    )
