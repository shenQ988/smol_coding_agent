"""/switch <name> — switch to an existing branch."""


def run(args: str = "", branch_manager=None, **kwargs) -> str:
    name = args.strip()
    if not name:
        return "Usage: /switch <name>"
    if branch_manager is None:
        return "Branch manager not available."

    try:
        thread_id = branch_manager.switch(name)
    except ValueError as e:
        return str(e)

    branch_manager.set_active(name)
    return f"__SWITCH_BRANCH__{thread_id}:{name}\nSwitched to branch '{name}'."
