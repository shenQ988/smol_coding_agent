"""/memory — show the agent's distilled memory for the current thread."""

def run(graph=None, graph_config=None, **kwargs) -> str:
    if graph is None or graph_config is None:
        return "No active session."

    current_state = graph.get_state(graph_config)
    memory = current_state.values.get("memory", {})

    return (
        f"Memory:\n"
        f"  task: {memory.get('task', '-')}\n"
        f"  files: {', '.join(memory.get('files', [])) or '-'}\n"
        f"  notes: {'; '.join(memory.get('notes', [])) or '-'}"
    )