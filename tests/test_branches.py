"""Phase 4a — BranchManager unit tests.

No graph execution needed here; these are pure BranchManager state tests.
"""

import json
import re
import time

import pytest

from agent.branches import BranchManager


@pytest.fixture
def bm(tmp_path):
    return BranchManager(index_path=tmp_path / ".branches.json")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_creates_main_branch(bm):
    branches = {b["name"] for b in bm.list_branches()}
    assert "main" in branches


def test_init_main_points_to_default_thread(bm):
    tid = bm.switch("main")
    assert tid == "default"


def test_init_persists_to_json(tmp_path):
    BranchManager(index_path=tmp_path / ".branches.json")
    assert (tmp_path / ".branches.json").exists()
    data = json.loads((tmp_path / ".branches.json").read_text())
    assert "main" in data["branches"]


# ---------------------------------------------------------------------------
# create_branch
# ---------------------------------------------------------------------------

def test_create_branch_returns_unique_thread_id(bm):
    tid1 = bm.create_branch("feat-a", from_thread_id="default")
    tid2 = bm.create_branch("feat-b", from_thread_id="default")
    assert tid1 != tid2


def test_create_branch_thread_id_format(bm):
    tid = bm.create_branch("auth", from_thread_id="default")
    assert tid.startswith("branch_auth_")


def test_create_branch_records_parent(bm):
    bm.set_active("main")
    bm.create_branch("child", from_thread_id="default")
    assert bm.get_parent("child") == "main"


def test_create_branch_persists(bm, tmp_path):
    bm.create_branch("persistent", from_thread_id="default")
    # Re-load from disk
    bm2 = BranchManager(index_path=tmp_path / ".branches.json")
    names = {b["name"] for b in bm2.list_branches()}
    assert "persistent" in names


def test_create_branch_duplicate_raises(bm):
    bm.create_branch("dup", from_thread_id="default")
    with pytest.raises(ValueError, match="already exists"):
        bm.create_branch("dup", from_thread_id="default")


# ---------------------------------------------------------------------------
# switch
# ---------------------------------------------------------------------------

def test_switch_returns_thread_id(bm):
    bm.create_branch("exp", from_thread_id="default")
    tid = bm.switch("exp")
    assert tid.startswith("branch_exp_")


def test_switch_nonexistent_raises_value_error(bm):
    """commands/switch.py catches ValueError by name — verify exact exception type."""
    with pytest.raises(ValueError, match="not found"):
        bm.switch("no-such-branch")


def test_switch_does_not_change_active_branch(bm):
    bm.create_branch("side", from_thread_id="default")
    bm.set_active("main")
    bm.switch("side")           # just retrieves thread_id
    assert bm.get_active() == "main"


# ---------------------------------------------------------------------------
# set_active / get_active
# ---------------------------------------------------------------------------

def test_set_active_changes_active(bm):
    bm.create_branch("new", from_thread_id="default")
    bm.set_active("new")
    assert bm.get_active() == "new"


def test_set_active_unknown_raises(bm):
    with pytest.raises(ValueError, match="not found"):
        bm.set_active("ghost")


def test_set_active_persists(bm, tmp_path):
    bm.create_branch("br", from_thread_id="default")
    bm.set_active("br")
    bm2 = BranchManager(index_path=tmp_path / ".branches.json")
    assert bm2.get_active() == "br"


# ---------------------------------------------------------------------------
# list_branches
# ---------------------------------------------------------------------------

def test_list_branches_marks_active(bm):
    bm.create_branch("x", from_thread_id="default")
    bm.set_active("x")
    listing = bm.list_branches()
    active_entries = [b for b in listing if b["active"]]
    assert len(active_entries) == 1
    assert active_entries[0]["name"] == "x"


def test_list_branches_contains_all(bm):
    bm.create_branch("a", from_thread_id="default")
    bm.create_branch("b", from_thread_id="default")
    names = {b["name"] for b in bm.list_branches()}
    assert names == {"main", "a", "b"}


# ---------------------------------------------------------------------------
# get_parent
# ---------------------------------------------------------------------------

def test_get_parent_main_is_none(bm):
    assert bm.get_parent("main") is None


def test_get_parent_unknown_is_none(bm):
    assert bm.get_parent("nonexistent") is None


def test_get_parent_nested(bm):
    bm.create_branch("child", from_thread_id="default")
    bm.set_active("child")
    bm.create_branch("grandchild", from_thread_id=bm.switch("child"))
    assert bm.get_parent("grandchild") == "child"
