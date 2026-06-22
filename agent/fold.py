"""Context folding — summarise a branch and return its outcome to the parent."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.branches import BranchManager
from memory.reduction import summarize_messages


def fold_branch(graph, branch_manager: BranchManager, llm, branch_name: str) -> str:
    """
    Summarise *branch_name*, append the summary to its parent thread, and
    switch the active branch back to the parent.

    Returns a human-readable confirmation string describing what was folded.
    """
    parent_name = branch_manager.get_parent(branch_name)
    if parent_name is None:
        return f"Cannot fold '{branch_name}' — it has no parent branch."

    branch_tid = branch_manager.switch(branch_name)
    parent_tid = branch_manager.switch(parent_name)

    # Load the branch's checkpointed messages
    branch_config = {"configurable": {"thread_id": branch_tid}}
    branch_state = graph.get_state(branch_config)
    messages = branch_state.values.get("messages", []) if branch_state else []

    if not messages:
        summary_text = "(no messages recorded in this branch)"
    else:
        try:
            summary_text = summarize_messages(messages, llm)
        except Exception as e:
            return f"Error: could not summarise branch '{branch_name}' — {e}"

    # Append the summary to the parent thread
    fold_message = HumanMessage(
        content=f"[Returning from branch '{branch_name}']\n{summary_text}"
    )
    parent_config = {"configurable": {"thread_id": parent_tid}}
    graph.update_state(parent_config, {"messages": [fold_message]})

    # Switch back to the parent
    branch_manager.set_active(parent_name)

    n = len(messages)
    return (
        f"Folded '{branch_name}' → '{parent_name}'. "
        f"Summarised {n} message{'s' if n != 1 else ''}.\n\n"
        f"Summary:\n{summary_text}"
    )
