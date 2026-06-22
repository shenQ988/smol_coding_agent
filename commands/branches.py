"""/branches — list all branches, marking the active one."""


def run(branch_manager=None, **kwargs) -> str:
    if branch_manager is None:
        return "Branch manager not available."

    branches = branch_manager.list_branches()
    if not branches:
        return "No branches found."

    lines = []
    for b in branches:
        marker = "*" if b["active"] else " "
        parent = f"  (parent: {b['parent']})" if b["parent"] else ""
        created = b["created_at"][:10]  # YYYY-MM-DD
        lines.append(f"  {marker} {b['name']:<24} {created}{parent}")

    return "Branches:\n" + "\n".join(lines)
