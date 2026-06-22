"""/fold — summarise the current branch and fold it back into its parent."""

from agent.fold import fold_branch as _fold_branch


def run(branch_manager=None, graph=None, llm=None, **kwargs) -> str:
    if branch_manager is None or graph is None or llm is None:
        return "Branch manager, graph, or LLM not available."

    active = branch_manager.get_active()
    if active == "main":
        return "Cannot fold 'main' — it has no parent branch."

    confirmation = _fold_branch(graph, branch_manager, llm, active)

    # After fold, branch_manager.get_active() is now the parent
    parent_name = branch_manager.get_active()
    parent_tid = branch_manager.switch(parent_name)
    return f"__SWITCH_BRANCH__{parent_tid}:{parent_name}\n{confirmation}"
