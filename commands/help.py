"""/help — list available commands."""

HELP_TEXT = """Commands:
  /exit    — quit the agent
  /help    — show this help
  /model   — show current model
  /memory  — show agent memory
  /clear   — clear conversation
  /cost    — show token usage and cost
  /compact — summarize old history to save context"""


def run(**kwargs) -> str:
    return HELP_TEXT
