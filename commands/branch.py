"""/branch <name> — create a new branch from the current thread and switch to it."""


def run(args: str = "", branch_manager=None, graph_config=None, **kwargs) -> str:
    name = args.strip()
    if not name:
        return "Usage: /branch <name>"
    if branch_manager is None:
        return "Branch manager not available."

    current_thread_id = graph_config["configurable"]["thread_id"]
    try:
        new_tid = branch_manager.create_branch(name, from_thread_id=current_thread_id)
    except ValueError as e:
        return str(e)

    branch_manager.set_active(name)
    return f"__SWITCH_BRANCH__{new_tid}:{name}\nCreated and switched to branch '{name}'."
