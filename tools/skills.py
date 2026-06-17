# tools/skills_tool.py
from langchain_core.tools import tool
from context.skills import SkillStore

_skill_store: SkillStore | None = None

def set_skill_store(store: SkillStore):
    global _skill_store
    _skill_store = store

@tool
def load_skill(name: str) -> str:
    """Load a skill by name to get specialized instructions for a task.

    Args:
        name: The skill name from the available skills list.
    """
    if _skill_store is None:
        return "Error: no skills configured"
    content = _skill_store.get(name)
    if content:
        return content
    return f"Error: skill '{name}' not found"