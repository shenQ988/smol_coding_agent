"""
BranchManager — tracks conversation branches for the agent.

Usage example:
    # 1. Agent starts; BranchManager auto-creates "main" pointing at thread "default"
    bm = BranchManager()

    # 2. User runs /branch auth-refactor  (currently on "main", thread "default")
    new_tid = bm.create_branch("auth-refactor", from_thread_id="default")
    bm.set_active("auth-refactor")
    # → agent now streams into thread "branch_auth-refactor_<ts>"

    # 3. User works in the branch — edits files, runs tests, etc.

    # 4. User runs /fold
    # fold_branch() summarises the branch's messages, appends the summary to "main",
    # then calls bm.set_active("main") — returning the agent to the parent thread.
    bm.get_active()  # → "main"
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


_INDEX_SCHEMA: dict[str, Any] = {
    "active": "main",
    "branches": {},
}


class BranchManager:
    """Persists a JSON index of agent branches and the currently active one."""

    def __init__(self, index_path: Path = Path(".branches.json")) -> None:
        self.index_path = index_path
        self._data: dict[str, Any] = {}
        self._load()
        if "main" not in self._data["branches"]:
            self._data["branches"]["main"] = {
                "thread_id": "default",
                "parent": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.index_path.exists():
            with open(self.index_path) as f:
                self._data = json.load(f)
        else:
            self._data = {"active": "main", "branches": {}}

    def _save(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── public API ────────────────────────────────────────────────────────────

    def create_branch(self, name: str, from_thread_id: str) -> str:
        """Create a new branch from *from_thread_id* and return its thread_id."""
        if name in self._data["branches"]:
            raise ValueError(f"Branch '{name}' already exists.")
        # Derive the parent name from the currently active branch
        parent_name = self._data["active"]
        thread_id = f"branch_{name}_{int(time.time() * 1000)}"
        self._data["branches"][name] = {
            "thread_id": thread_id,
            "parent": parent_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        return thread_id

    def switch(self, name: str) -> str:
        """Return the thread_id for *name*, raising ValueError if not found."""
        entry = self._data["branches"].get(name)
        if entry is None:
            raise ValueError(f"Branch '{name}' not found.")
        return entry["thread_id"]

    def list_branches(self) -> list[dict]:
        """Return all branches with metadata, marking the currently active one."""
        active = self._data["active"]
        result = []
        for name, meta in self._data["branches"].items():
            result.append({
                "name": name,
                "thread_id": meta["thread_id"],
                "parent": meta["parent"],
                "created_at": meta["created_at"],
                "active": name == active,
            })
        return result

    def set_active(self, name: str) -> None:
        """Mark *name* as the currently active branch and persist."""
        if name not in self._data["branches"]:
            raise ValueError(f"Branch '{name}' not found.")
        self._data["active"] = name
        self._save()

    def get_active(self) -> str:
        """Return the currently active branch name."""
        return self._data["active"]

    def get_parent(self, name: str) -> str | None:
        """Return the parent branch name for *name*, or None for root branches."""
        entry = self._data["branches"].get(name)
        return entry["parent"] if entry else None
