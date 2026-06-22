from commands import (
    branch, branches, compact, cost, exit as exit_cmd,
    fold, help as help_cmd, memory, model, clear, switch,
)

COMMANDS = {
    "/branch":   branch.run,
    "/branches": branches.run,
    "/clear":    clear.run,
    "/compact":  compact.run,
    "/cost":     cost.run,
    "/exit":     exit_cmd.run,
    "/fold":     fold.run,
    "/help":     help_cmd.run,
    "/memory":   memory.run,
    "/model":    model.run,
    "/switch":   switch.run,
}


def dispatch(command: str, **context) -> str | None:
    """
    Dispatch a slash command string to its handler.

    Splits on the first space so that '/branch auth-refactor' routes to the
    '/branch' handler with args='auth-refactor'.  Existing no-arg commands
    receive args='' and ignore it via **kwargs.
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""
    handler = COMMANDS.get(cmd)
    if handler is None:
        return None
    return handler(args=args, **context)
