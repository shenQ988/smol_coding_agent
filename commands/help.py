"""/help — list available commands."""

HELP_TEXT = """Commands:
  /exit              — quit the agent
  /help              — show this help
  /model             — show current provider and model
  /memory            — show agent memory for the current thread
  /clear             — start a fresh conversation thread
  /cost              — show token usage and estimated cost
  /compact           — summarize old history to save context
  /branch <name>     — create a new branch from the current thread
  /switch <name>     — switch to an existing branch
  /branches          — list all branches
  /fold              — summarize current branch and return to parent"""


def run(**kwargs) -> str:
    return HELP_TEXT
