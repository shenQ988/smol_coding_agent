"""/exit — quit the agent. Handled specially in main.py's loop (breaks out),
but defined here for consistency with /help listing."""

def run(**kwargs) -> str:
    return "__EXIT__"   # sentinel value main.py checks for
