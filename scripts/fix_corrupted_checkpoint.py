"""
One-off script: remove any SystemMessages stored in the 'default' thread checkpoint.

The think node prepends its own SystemMessage fresh each turn and never persists it,
so there should be ZERO SystemMessages in the checkpoint. Any that exist are corrupted
leftovers from a prior /fold bug (which injected SystemMessage instead of HumanMessage).
"""

import sqlite3
import sys
from pathlib import Path

# Make sure imports resolve from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import RemoveMessage
from langchain_core.messages import SystemMessage

from agent.graph import build_graph

THREAD_ID = "default"
DB_PATH = str(Path(__file__).parent.parent / ".checkpoints.db")


def main():
    print(f"Connecting to {DB_PATH} ...")
    graph, db_conn = build_graph(provider="anthropic", model="claude-sonnet-4-6", db_path=DB_PATH)

    try:
        config = {"configurable": {"thread_id": THREAD_ID}}
        state = graph.get_state(config)

        if not state or not state.values:
            print("No checkpoint found for thread 'default'. Nothing to fix.")
            return

        messages = state.values.get("messages", [])
        print(f"Total messages in checkpoint: {len(messages)}")

        # Find SystemMessages at any index (all of them are wrong in checkpoint storage)
        bad = [(i, m) for i, m in enumerate(messages) if isinstance(m, SystemMessage)]

        if not bad:
            print("No SystemMessages found in checkpoint — thread is already clean.")
            return

        print(f"\nFound {len(bad)} SystemMessage(s) to remove:")
        for i, m in bad:
            preview = (m.content[:120] + "...") if len(m.content) > 120 else m.content
            print(f"  [{i}] id={m.id!r}  content={preview!r}")

        removals = [RemoveMessage(id=m.id) for _, m in bad]
        graph.update_state(config, {"messages": removals})
        print(f"\nRemoved {len(removals)} message(s).")

        # Verify
        state2 = graph.get_state(config)
        remaining = state2.values.get("messages", []) if state2 else []
        still_bad = [m for m in remaining if isinstance(m, SystemMessage)]
        if still_bad:
            print(f"WARNING: {len(still_bad)} SystemMessage(s) still remain — manual inspection needed.")
        else:
            print(f"Thread 'default' is clean. {len(remaining)} message(s) remain.")

    finally:
        db_conn.close()


if __name__ == "__main__":
    main()
