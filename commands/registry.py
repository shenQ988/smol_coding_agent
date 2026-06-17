from commands import compact, cost, help as help_cmd, model, memory, clear, exit as exit_cmd

COMMANDS = {
    "/help": help_cmd.run,
    "/model": model.run,
    "/memory": memory.run,
    "/clear": clear.run,
    "/cost": cost.run,
    "/compact": compact.run,
    "/exit": exit_cmd.run,
}


def dispatch(command: str, **context) -> str | None:
    handler = COMMANDS.get(command)
    if handler is None:
        return None
    return handler(**context)