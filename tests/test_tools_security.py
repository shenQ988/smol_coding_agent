"""Phase 6 — Path sandboxing / workspace containment tests.

Every filesystem tool must refuse paths that escape WORKSPACE_ROOT.
The 'search' tool returns an error string; the filesystem tools raise ValueError.
Both behaviours are tested both at the tool level and through act() which
converts all exceptions to error ToolMessages (so no exception leaks to the REPL).
"""

import pytest
from langchain_core.messages import ToolMessage

from tools import filesystem, search as search_module
from tools.filesystem import list_files, patch_file, read_file, write_file
from tools.search import search
from agent.nodes import act
from tests.conftest import ai_with_tool, make_state


TRAVERSAL_PATHS = [
    "../../etc/passwd",
    "../../../etc/shadow",
    "../../../../../../etc/hosts",
]

ABSOLUTE_PATHS = [
    "/etc/passwd",
    "/tmp/evil",
    "/var/log/system.log",
]


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_workspace, monkeypatch):
    """Point all tool WORKSPACE_ROOT globals at tmp_workspace for every test."""
    monkeypatch.setattr(filesystem, "WORKSPACE_ROOT", tmp_workspace)
    monkeypatch.setattr(search_module, "WORKSPACE_ROOT", tmp_workspace)
    return tmp_workspace


# ---------------------------------------------------------------------------
# read_file — raises ValueError for escaping paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", TRAVERSAL_PATHS + ABSOLUTE_PATHS)
def test_read_file_blocks_path_escape(path):
    with pytest.raises(ValueError, match="[Ee]scapes? workspace"):
        read_file.invoke({"path": path})


# ---------------------------------------------------------------------------
# write_file — raises ValueError for escaping paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", TRAVERSAL_PATHS + ABSOLUTE_PATHS)
def test_write_file_blocks_path_escape(path):
    with pytest.raises(ValueError, match="[Ee]scapes? workspace"):
        write_file.invoke({"path": path, "content": "malicious"})


# ---------------------------------------------------------------------------
# patch_file — raises ValueError for escaping paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", TRAVERSAL_PATHS + ABSOLUTE_PATHS)
def test_patch_file_blocks_path_escape(path):
    with pytest.raises(ValueError, match="[Ee]scapes? workspace"):
        patch_file.invoke({"path": path, "old_text": "a", "new_text": "b"})


# ---------------------------------------------------------------------------
# list_files — raises ValueError for escaping paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", TRAVERSAL_PATHS + ABSOLUTE_PATHS)
def test_list_files_blocks_path_escape(path):
    with pytest.raises(ValueError, match="[Ee]scapes? workspace"):
        list_files.invoke({"path": path})


# ---------------------------------------------------------------------------
# search — returns error string (not raise) — regression guard for the bug
# we found where search was missing sandboxing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", TRAVERSAL_PATHS + ABSOLUTE_PATHS)
def test_search_blocks_path_escape(path):
    """search returns 'Error: path escapes workspace' rather than searching outside."""
    result = search.invoke({"pattern": "root", "path": path})
    assert "Error" in result
    assert "escapes workspace" in result


# ---------------------------------------------------------------------------
# Safe paths — must NOT be blocked
# ---------------------------------------------------------------------------

def test_read_file_allows_workspace_path(tmp_workspace):
    """A legitimate workspace-relative path must succeed (not be over-blocked)."""
    (tmp_workspace / "hello.txt").write_text("hello")
    result = read_file.invoke({"path": "hello.txt"})
    assert "hello" in result


def test_list_files_allows_root(tmp_workspace):
    """Listing '.' (workspace root) must succeed."""
    result = list_files.invoke({"path": "."})
    assert "Error" not in result or "does not exist" not in result


def test_search_allows_workspace_root(tmp_workspace):
    """Searching '.' must not be blocked by the sandbox."""
    (tmp_workspace / "code.py").write_text("def foo(): pass")
    result = search.invoke({"pattern": "def foo", "path": "."})
    # Should find the pattern, not return an escape error
    assert "escapes workspace" not in result


# ---------------------------------------------------------------------------
# act() integration — all escaping paths produce error ToolMessages
# (no exception propagates to the REPL; act catches everything)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,args,tool_obj", [
    ("read_file",  {"path": "../../etc/passwd"}, read_file),
    ("list_files", {"path": "../../etc"},        list_files),
    # write_file and patch_file are RISKY_TOOLS — act() interrupts before reaching
    # the tool, so they can't be tested this way without a full graph context.
    # Their sandbox is verified by the direct-tool tests above.
])
def test_act_converts_sandbox_violation_to_tool_message(tool_name, args, tool_obj):
    """act() wraps sandbox ValueError in an error ToolMessage — no raise to REPL."""
    state = make_state(messages=[ai_with_tool(tool_name, args)])
    result = act(state, tool_map={tool_name: tool_obj})
    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], ToolMessage)
    assert "error" in msgs[0].content.lower()
